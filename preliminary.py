import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==========================================
# SHARED UTILITIES
# ==========================================
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape for RoPE: [B, T, H, D] -> [B, T, H, D/2, 2]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].view(1, xq.shape[1], 1, -1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.g

# ==========================================
# 1. STANDARD ATTENTION (Baseline - O(T^2))
# ==========================================
class StandardAttention(nn.Module):
    def __init__(self, d_model, n_head, **kwargs):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

# ==========================================
# 2. HYBRID COMPONENT: Causal FFT Conv
# ==========================================
class CausalFFTConv(nn.Module):
    def __init__(self, d_model, max_freq=128):
        super().__init__()
        self.d_model = d_model
        # Learnable decay and frequency params
        self.decay = nn.Parameter(torch.randn(d_model))
        self.freq = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        B, T, D = x.shape
        # 1. Construct Filter in Time Domain
        t = torch.arange(T, device=x.device).float()
        k = torch.exp(-torch.abs(self.decay.view(1, -1)) * t.view(-1, 1)) * \
            torch.cos(self.freq.view(1, -1) * t.view(-1, 1)) 
        
        # 2. Causal Padding (Pad to 2T to avoid circular convolution artifacts)
        n_fft = 2 * T
        
        x_f = torch.fft.rfft(x, n=n_fft, dim=1) 
        k_f = torch.fft.rfft(k, n=n_fft, dim=0) 
        
        # 3. Convolution
        y_f = x_f * k_f.unsqueeze(0)
        
        # 4. Inverse FFT & Crop
        y = torch.fft.irfft(y_f, n=n_fft, dim=1)
        return y[:, :T, :] 

# ==========================================
# 3. HYBRID COMPONENT: Chunked Attention (O(T))
# ==========================================
class ChunkedSlidingWindowAttention(nn.Module):
    """
    Implements Sliding Window Attention by splitting T into Chunks.
    Complexity: O(T * WindowSize) -> Linear in T.
    Memory: O(T * WindowSize) -> Linear in T.
    """
    def __init__(self, d_model, n_head, window_size=32):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis=None):
        B, T, C = x.shape
        W = self.window_size
        
        # 1. Pad T to be divisible by Window Size W
        pad_len = (W - (T % W)) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len)) # Pad time dim
        
        B, T_pad, _ = x.shape
        n_chunks = T_pad // W

        # 2. Projections [B, T_pad, n_head, head_dim]
        q = self.q_proj(x).view(B, T_pad, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T_pad, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T_pad, self.n_head, self.head_dim)

        if freqs_cis is not None:
            # Handle RoPE padding if necessary
            if freqs_cis.shape[0] < T_pad:
                 # Extend freqs_cis dynamically if T > precomputed
                 freqs_cis = precompute_freqs_cis(self.head_dim, T_pad * 2).to(x.device)
            q, k = apply_rotary_emb(q, k, freqs_cis[:T_pad])

        # 3. Chunking [B, n_chunks, W, n_head, head_dim]
        q_chunks = q.view(B, n_chunks, W, self.n_head, self.head_dim)
        k_chunks = k.view(B, n_chunks, W, self.n_head, self.head_dim)
        v_chunks = v.view(B, n_chunks, W, self.n_head, self.head_dim)
        
        # 4. Construct Context (Prev Chunk + Current Chunk)
        # Shift chunks to right to get "previous"
        k_prev = torch.roll(k_chunks, shifts=1, dims=1)
        v_prev = torch.roll(v_chunks, shifts=1, dims=1)
        
        # Zero out the "previous" of the 0-th chunk
        k_prev[:, 0] = 0
        v_prev[:, 0] = 0
        
        # Concat: [B, n_chunks, 2*W, n_head, head_dim]
        k_context = torch.cat([k_prev, k_chunks], dim=2)
        v_context = torch.cat([v_prev, v_chunks], dim=2)
        
        # 5. Attention Computation
        # Reshape to: [B * n_chunks, n_head, W, head_dim]
        q_flat = q_chunks.permute(0, 1, 3, 2, 4).reshape(-1, self.n_head, W, self.head_dim)
        # Keys have length 2*W
        k_flat = k_context.permute(0, 1, 3, 2, 4).reshape(-1, self.n_head, 2*W, self.head_dim)
        v_flat = v_context.permute(0, 1, 3, 2, 4).reshape(-1, self.n_head, 2*W, self.head_dim)

        # Q @ K.T -> [..., W, 2*W]
        scores = (q_flat @ k_flat.transpose(-2, -1)) * self.scale
        
        # --- Masking ---
        # 1. Inner Chunk Mask (Causal for current chunk part)
        # Shape [W, 2W]. 
        # Left side [:, :W] is all visible (previous chunk).
        # Right side [:, W:] is causal diagonal.
        mask_matrix = torch.ones(W, 2*W, device=x.device)
        mask_matrix[:, W:] = torch.tril(torch.ones(W, W, device=x.device))
        
        scores = scores.masked_fill(mask_matrix == 0, float('-inf'))
        
        # 2. Boundary Mask (Chunk 0 has no previous chunk)
        # Reshape to expose chunk dimension
        scores = scores.view(B, n_chunks, self.n_head, W, 2*W)
        # Mask the "previous" part (indices 0..W) for chunk 0
        scores[:, 0, :, :, :W] = float('-inf')
        
        # Flatten back
        scores = scores.view(-1, self.n_head, W, 2*W)

        attn = F.softmax(scores, dim=-1)
        out_flat = (attn @ v_flat) # [..., W, head_dim]
        
        # 6. Reassemble
        out = out_flat.view(B, n_chunks, self.n_head, W, self.head_dim)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T_pad, -1) # Flatten
        
        # Remove padding
        if pad_len > 0:
            out = out[:, :T, :]
            
        return self.out_proj(out)

class AMO_V2_Efficient(nn.Module):
    def __init__(self, d_model, n_head, **kwargs):
        super().__init__()
        # Branch 1: FFT Conv (Global)
        self.conv_proj = nn.Linear(d_model, d_model)
        self.conv = CausalFFTConv(d_model)
        self.conv_norm = RMSNorm(d_model) 
        
        # Branch 2: Chunked Attention (Local, Linear Complexity)
        self.attn = ChunkedSlidingWindowAttention(d_model, n_head, window_size=32)
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, freqs_cis=None):
        conv_out = self.conv_norm(self.conv(self.conv_proj(x)))
        attn_out = self.attn(x, freqs_cis)
        return self.out_proj(conv_out + attn_out)

# ==========================================
# BACKBONE & GYM
# ==========================================
class Transformer(nn.Module):
    def __init__(self, attn_cls, vocab, d_model, n_head, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        # Precompute RoPE for max expected length (Needle test uses 256+)
        self.freqs_cis = precompute_freqs_cis(d_model // n_head, 4096).to(device)
            
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': RMSNorm(d_model),
                'attn': attn_cls(d_model, n_head),
                'norm2': RMSNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
            }) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x)
        freqs = self.freqs_cis[:T]

        for l in self.layers:
            x = x + l['attn'](l['norm1'](x), freqs_cis=freqs)
            x = x + l['ffn'](l['norm2'](x))
        return self.head(self.final_norm(x))

class TaskGym:
    def __init__(self, vocab=50, batch=64):
        self.vocab = vocab
        self.batch = batch
        
    def get_associative(self, seq_len):
        data = torch.randint(1, self.vocab, (self.batch, seq_len)).to(device)
        targets = torch.randint(0, self.vocab, (self.batch,)).to(device)
        for i in range(self.batch):
            k, v = torch.randint(1, self.vocab, (2,)).tolist()
            pos = torch.randint(0, seq_len // 2, (1,)).item()
            data[i, pos] = k
            data[i, pos+1] = v
            data[i, -1] = k
            targets[i] = v
        return data, targets

    def get_induction(self, seq_len):
        data = torch.randint(1, self.vocab, (self.batch, seq_len)).to(device)
        targets = torch.zeros(self.batch, dtype=torch.long).to(device)
        for i in range(self.batch):
            k = torch.randint(1, self.vocab, (1,)).item()
            v = torch.randint(1, self.vocab, (1,)).item()
            pos = torch.randint(0, seq_len - 10, (1,)).item()
            data[i, pos] = k
            data[i, pos+1] = v
            data[i, -1] = k
            targets[i] = v
        return data, targets
    
    def get_sorting(self, seq_len):
        data = torch.randint(1, self.vocab, (self.batch, seq_len)).to(device)
        targets = torch.max(data, dim=1)[0]
        data[:, -1] = 0
        return data, targets

    def get_needle(self, seq_len):
        # Place a needle in a long haystack
        data = torch.randint(1, self.vocab, (self.batch, seq_len)).to(device)
        targets = torch.zeros(self.batch, dtype=torch.long).to(device)
        for i in range(self.batch):
            k, v = torch.randint(1, self.vocab, (2,)).tolist()
            # Random position in first 80%
            pos = torch.randint(0, int(seq_len * 0.8), (1,)).item()
            data[i, pos] = k
            data[i, pos+1] = v
            data[i, -1] = k
            targets[i] = v
        return data, targets

# ==========================================
# EXPERIMENT RUNNER
# ==========================================
def run_suite():
    methods = {
        "Standard": StandardAttention,
        "Hybrid": AMO_V2_Efficient,
    }
    
    gym = TaskGym(vocab=50, batch=64)
    criterion = nn.CrossEntropyLoss()
    
    # --- CONFIGURATION ---
    d_model = 128
    n_head = 4
    n_layers = 2
    STEPS = 3000 
    
    train_tasks = ["Associative", "Induction", "Sorting"]
    metrics_list = ["Associative", "Induction", "Sorting", "LenGen(4x)", "Needle"]
    
    results = {m: {t: 0.0 for t in metrics_list} for m in methods}
    
    print(f"{'Method':<14} | {'Training Task':<15} | {'Loss':<6} | {'Acc':<6}")
    print("-" * 65)

    for task_name in train_tasks:
        for method_name, cls in methods.items():
            model = Transformer(cls, vocab=50, d_model=d_model, n_head=n_head, n_layers=n_layers).to(device)
            
            lr = 1e-3
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=STEPS, pct_start=0.2)
            
            model.train()
            losses = []
            
            # Training Loop
            for s in range(STEPS):
                if task_name == "Associative": x, y = gym.get_associative(32)
                elif task_name == "Induction": x, y = gym.get_induction(32)
                elif task_name == "Sorting": x, y = gym.get_sorting(32)
                
                optimizer.zero_grad()
                out = model(x)[:, -1, :]
                loss = criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                losses.append(loss.item())
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                # Base Task Accuracy
                pred = out.argmax(-1)
                acc = (pred == y).float().mean().item()
                results[method_name][task_name] = acc
                
                print(f"{method_name:<14} | {task_name:<15} | {np.mean(losses[-100:]):.3f}  | {acc:.2f}")
                
                # Generalization Tests (Relevant for Associative/Retrieval tasks)
                if task_name == "Associative":
                    # LenGen: Train 32 -> Test 128 (4x)
                    x_lg, y_lg = gym.get_associative(128)
                    acc_lg = (model(x_lg)[:, -1, :].argmax(-1) == y_lg).float().mean().item()
                    results[method_name]["LenGen(4x)"] = acc_lg
                    
                    # Needle: Train 32 -> Test 256 (Extreme length generalization)
                    # This tests if the Convolution holds up and if sliding window doesn't break
                    x_nd, y_nd = gym.get_needle(256)
                    acc_nd = (model(x_nd)[:, -1, :].argmax(-1) == y_nd).float().mean().item()
                    results[method_name]["Needle"] = acc_nd

    print("\nGenerating Final Report...")
    
    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metrics_list))
    width = 0.35
    
    for i, (m_name, m_res) in enumerate(results.items()):
        vals = [m_res[k] for k in metrics_list]
        ax.bar(x + (i - 0.5)*width, vals, width, label=m_name)
        
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Comparison: Standard vs Hybrid (Efficient O(T) Chunking) - Steps={STEPS}")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n=== FINAL METRICS TABLE ===")
    header = f"{'Method':<14} " + " ".join([f"{t:<12}" for t in metrics_list])
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:<14} " + " ".join([f"{results[m][t]:<12.2f}" for t in metrics_list])
        print(row)

if __name__ == "__main__":
    run_suite()