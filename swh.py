# ==========================================
# 0. SETUP & DEPENDENCIES
# ==========================================
import os
import gc
import time
import math
import matplotlib.pyplot as plt
import numpy as np

# Install dependencies (uncomment if needed in a fresh kernel)
# os.system('pip install -q pytorch-lightning datasets tiktoken transformers matplotlib')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from datasets import load_dataset
import tiktoken

# --- EXPERIMENT CONFIGURATION ---
DEBUG = False #True 

CONFIG = {
    'seed': 1337,
    'block_size': 1024,      
    'vocab_size': 50304,     
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'dropout': 0.0,
    
    # BATCH SIZE MATH:
    # BS=4 per GPU * 2 GPUs = 8 Physical
    'batch_size': 4,         
    'grad_accum_steps': 8,
    
    # DEBUG SETTING:
    # 5 iters * 8 accum = 40 batches. This is enough to test sanity without waiting too long.
    'max_iters': 5 if DEBUG else 2000, 
    
    'lr': 6e-4,
    'weight_decay': 1e-1,

    'num_workers': 0 
}

pl.seed_everything(CONFIG['seed'], workers=True)
torch.set_float32_matmul_precision('medium') 

# ==========================================
# 1. DATASET (Network Safe)
# ==========================================
class FineWebIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.dataset = None 

    def __iter__(self):
        # Workers initialize dataset locally
        if self.dataset is None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to load dataset after {max_retries} attempts.")
                        raise e
                    time.sleep(2) # Wait before retry
            
        iterator = iter(self.dataset)
        buffer = []
        while True:
            try:
                while len(buffer) < (self.batch_size * self.block_size + 1):
                    text = next(iterator)['text']
                    buffer.extend(self.enc.encode(text))
                
                data = torch.tensor(buffer[:self.batch_size * self.block_size + 1], dtype=torch.long)
                buffer = buffer[self.batch_size * self.block_size:] 
                
                x = data[:-1].view(self.batch_size, self.block_size)
                y = data[1:].view(self.batch_size, self.block_size)
                yield x, y
            except StopIteration:
                iterator = iter(self.dataset) 

# ==========================================
# 2. MODELS (Unchanged - Integrity Preserved)
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].view(1, xq.shape[1], 1, -1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLU(nn.Module):
    def __init__(self, dim_in, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim_in, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim_in, bias=False)
        self.w3 = nn.Linear(dim_in, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalFFTConv(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.decay = nn.Parameter(torch.randn(d_model))
        self.freq = nn.Parameter(torch.randn(d_model))
    
    def forward(self, x):
        B, T, D = x.shape
        t = torch.arange(T, device=x.device).float()
        
        # Compute Kernel
        k = torch.exp(-torch.abs(self.decay.view(1, -1)) * t.view(-1, 1)) * \
            torch.cos(self.freq.view(1, -1) * t.view(-1, 1))
        
        n_fft = 2 * T
        
        x_32 = x.to(torch.float32)
        k_32 = k.to(torch.float32)
        
        x_f = torch.fft.rfft(x_32, n=n_fft, dim=1) 
        k_f = torch.fft.rfft(k_32, n=n_fft, dim=0) 
        
        y = torch.fft.irfft(x_f * k_f.unsqueeze(0), n=n_fft, dim=1)
        
        # Cast back to original type (float16/bfloat16)
        return y[:, :T, :].to(x.dtype)

class ChunkedSlidingWindowAttention(nn.Module):
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
        self.register_buffer("mask_matrix", torch.tril(torch.ones(window_size, window_size)))

    def forward(self, x, freqs_cis=None):
        B, T, C = x.shape
        W = self.window_size
        pad_len = (W - (T % W)) % W
        if pad_len > 0: x = F.pad(x, (0, 0, 0, pad_len))
        B, T_pad, _ = x.shape
        n_chunks = T_pad // W
        q = self.q_proj(x).view(B, T_pad, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T_pad, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T_pad, self.n_head, self.head_dim)
        if freqs_cis is not None:
             if freqs_cis.shape[0] < T_pad:
                 freqs_cis = precompute_freqs_cis(self.head_dim, T_pad * 2).to(x.device)
             q, k = apply_rotary_emb(q, k, freqs_cis[:T_pad])
        q_chunks = q.view(B, n_chunks, W, self.n_head, self.head_dim)
        k_chunks = k.view(B, n_chunks, W, self.n_head, self.head_dim)
        v_chunks = v.view(B, n_chunks, W, self.n_head, self.head_dim)
        k_prev = torch.roll(k_chunks, shifts=1, dims=1); v_prev = torch.roll(v_chunks, shifts=1, dims=1)
        k_prev[:, 0] = 0; v_prev[:, 0] = 0
        k_context = torch.cat([k_prev, k_chunks], dim=2)
        v_context = torch.cat([v_prev, v_chunks], dim=2)
        q_flat = q_chunks.permute(0, 1, 3, 2, 4).reshape(-1, self.n_head, W, self.head_dim)
        k_flat = k_context.permute(0, 1, 3, 2, 4).reshape(-1, self.n_head, 2*W, self.head_dim)
        v_flat = v_context.permute(0, 1, 3, 2, 4).reshape(-1, self.n_head, 2*W, self.head_dim)
        scores = (q_flat @ k_flat.transpose(-2, -1)) * self.scale
        full_mask = torch.ones(W, 2*W, device=x.device); full_mask[:, W:] = self.mask_matrix
        scores = scores.masked_fill(full_mask == 0, float('-inf'))
        scores = scores.view(B, n_chunks, self.n_head, W, 2*W); scores[:, 0, :, :, :W] = float('-inf')
        scores = scores.view(-1, self.n_head, W, 2*W)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v_flat).view(B, n_chunks, self.n_head, W, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, T_pad, -1)
        if pad_len > 0: out = out[:, :T, :]
        return self.out_proj(out)

class AMO_V2_Efficient(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.conv_proj = nn.Linear(d_model, d_model)
        self.conv = CausalFFTConv(d_model)
        self.conv_norm = RMSNorm(d_model)
        self.attn = ChunkedSlidingWindowAttention(d_model, n_head, window_size=32)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, freqs_cis):
        return self.out_proj(self.conv_norm(self.conv(self.conv_proj(x))) + self.attn(x, freqs_cis))

# --- BASELINE ---
class StandardAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)) == 0
        scores = scores.masked_fill(mask, float('-inf'))
        return self.out_proj((F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C))

# ==========================================
# 3. LIGHTNING MODULE
# ==========================================
class LlamaBlock(nn.Module):
    def __init__(self, config, attn_type):
        super().__init__()
        self.rms_1 = RMSNorm(config['n_embd'])
        self.attn = AMO_V2_Efficient(config['n_embd'], config['n_head']) if attn_type == 'Hybrid' else StandardAttention(config['n_embd'], config['n_head'])
        self.rms_2 = RMSNorm(config['n_embd'])
        self.ffn = SwiGLU(config['n_embd'], int(4 * config['n_embd'] * (2/3)))
    def forward(self, x, freqs_cis):
        return x + self.attn(self.rms_1(x), freqs_cis) + self.ffn(self.rms_2(x + self.attn(self.rms_1(x), freqs_cis)))

class LLMExperiment(pl.LightningModule):
    def __init__(self, config, attn_type='Baseline'):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.token_emb = nn.Embedding(config['vocab_size'], config['n_embd'])
        self.layers = nn.ModuleList([LlamaBlock(config, attn_type) for _ in range(config['n_layer'])])
        self.final_norm = RMSNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        freqs_complex = precompute_freqs_cis(config['n_embd'] // config['n_head'], config['block_size'] * 4)
        self.register_buffer('freqs_cis_real', torch.view_as_real(freqs_complex))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx)
        freqs_cis = torch.view_as_complex(self.freqs_cis_real[:T])
        
        for layer in self.layers:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, freqs_cis, use_reentrant=False)
            else:
                x = layer(x, freqs_cis)
        x = self.final_norm(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        hidden_states = self(x)
        loss = 0
        chunk_size = 256
        B, T, C = hidden_states.shape
        hidden_flat = hidden_states.view(-1, C)
        targets_flat = y.view(-1)
        
        for i in range(0, B * T, chunk_size):
            end = min(i + chunk_size, B * T)
            loss += F.cross_entropy(self.lm_head(hidden_flat[i:end]), targets_flat[i:end], reduction='sum')
            
        loss = loss / (B * T)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])

# ==========================================
# 4. CALLBACKS & BENCHMARKING
# ==========================================
class LossLoggerCallback(Callback):
    def __init__(self, filename):
        self.filename = filename
        self.losses = []
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0:
            # Safely extract loss
            try:
                loss_val = outputs['loss'].item()
            except (KeyError, AttributeError):
                # Fallback for accumulated steps where outputs might be None
                loss_val = trainer.callback_metrics.get('train_loss', 0).item()
                
            self.losses.append(loss_val)

    def on_train_end(self, trainer, pl_module):
        # Save to disk so the Main Process can read it later
        if trainer.global_rank == 0:
            np.save(self.filename, np.array(self.losses))
            print(f"Saved losses to {self.filename}")

def benchmark_model(pl_model, seq_lens):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = pl_model.to(device)
    model.eval()
    times, mems = [], []
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\nBenchmarking {model.__class__.__name__} Scaling...")
    
    for T in seq_lens:
        try:
            torch.cuda.empty_cache()
            x = torch.randint(0, CONFIG['vocab_size'], (1, T), device=device)
            with torch.no_grad(): _ = model(x) # Warmup
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                for _ in range(10): _ = model(x)
            torch.cuda.synchronize()
            avg_time = (time.time() - start) / 10
            mem = torch.cuda.max_memory_allocated() / 1024**2
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"SeqLen {T}: {avg_time*1000:.2f}ms | {mem:.0f}MB")
            times.append(avg_time); mems.append(mem)
        except RuntimeError:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"SeqLen {T}: OOM")
            times.append(None); mems.append(None)
    return times, mems

# ==========================================
# 5. RUNNER
# ==========================================
def run_experiment_phase(attn_type, filename):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"\n=== STARTING TRAINING: {attn_type} ===")
    
    ds = FineWebIterableDataset(CONFIG['batch_size'], CONFIG['block_size'])
    # Keep num_workers=0 for Kaggle stability
    train_loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    
    model = LLMExperiment(CONFIG, attn_type=attn_type)
    
    total_batches_needed = CONFIG['max_iters'] * CONFIG['grad_accum_steps']

    # 1. Define Checkpoint Callback (Saves model every 500 steps)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"{attn_type}-{{step}}",
        every_n_train_steps=500, 
        save_top_k=-1
    )
    
    # 2. Define Trainer with BOTH callbacks
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=2,
        strategy='ddp_notebook', 
        precision='16-mixed',
        max_steps=CONFIG['max_iters'],
        limit_train_batches=total_batches_needed,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        max_epochs=-1, 
        gradient_clip_val=1.0,
        accumulate_grad_batches=CONFIG['grad_accum_steps'],
        
        enable_checkpointing=True, 
        callbacks=[LossLoggerCallback(filename), checkpoint_callback],
        
        logger=False,
        enable_progress_bar=True
    )
    
    trainer.fit(model, train_loader)
    return model
    
if __name__ == "__main__":
    # Define filenames for persistence
    file_base = "loss_baseline.npy"
    file_hybr = "loss_hybrid.npy"

    # Run phases (Passing filenames, not lists)
    model_base = run_experiment_phase('Baseline', file_base)
    del model_base
    torch.cuda.empty_cache()
    gc.collect()
    
    model_hybr = run_experiment_phase('Hybrid', file_hybr)

    # --- REPORTING ---
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        baseline_losses = np.load(file_base) if os.path.exists(file_base) else []
        hybrid_losses = np.load(file_hybr) if os.path.exists(file_hybr) else []
        
        # PPL Calculation
        mean_base = np.mean(baseline_losses[-50:]) if len(baseline_losses) > 0 else 0
        mean_hybr = np.mean(hybrid_losses[-50:]) if len(hybrid_losses) > 0 else 0
        
        ppl_base = math.exp(mean_base) if mean_base > 0 else 0
        ppl_hybr = math.exp(mean_hybr) if mean_hybr > 0 else 0
        
        print(f"\nFinal PPL -> Baseline: {ppl_base:.2f} | Hybrid: {ppl_hybr:.2f}")

        # Plot Loss
        plt.figure(figsize=(10, 5))
        if len(baseline_losses) > 0:
            plt.plot(baseline_losses, label=f'Baseline {ppl_base:.2f}')
        if len(hybrid_losses) > 0:
            plt.plot(hybrid_losses, label=f'Hybrid {ppl_hybr:.2f}')
        plt.legend(); plt.grid(True); plt.savefig("loss_curve.png"); plt.show()

        # Benchmark
        model_base = LLMExperiment(CONFIG, attn_type='Baseline')
        model_hybr = LLMExperiment(CONFIG, attn_type='Hybrid')
        
        seq_lens = [512, 1024, 2048, 4096]
        t_base, m_base = benchmark_model(model_base, seq_lens)
        t_hybr, m_hybr = benchmark_model(model_hybr, seq_lens)
        
        # Plot Benchmark
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(seq_lens, t_base, 'o-', label='Baseline'); ax[0].plot(seq_lens, t_hybr, 'o-', label='Hybrid')
        ax[0].set_title("Latency (s)"); ax[0].legend()
        ax[1].plot(seq_lens, m_base, 'o-', label='Baseline'); ax[1].plot(seq_lens, m_hybr, 'o-', label='Hybrid')
        ax[1].set_title("VRAM (MB)"); ax[1].legend()
        plt.savefig("benchmark.png"); plt.show()

# # ---------------------------------------------------------
# # USE THIS CODE FOR PRODUCING SMOOTHED CURVES PRESENTED IN PAPER.
# # ---------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------
# # 1. Helper Function: Moving Average (Standard Rolling Mean)
# # ---------------------------------------------------------
# def moving_average(data, window_size=50):
#     """
#     Calculates the rolling average using numpy convolution.
#     This sits in the 'center' of the noise rather than lagging behind.
#     """
#     # Create a kernel (window) of ones normalized by the window size
#     kernel = np.ones(window_size) / window_size
#     # Convolve to get the average
#     # 'valid' means we only keep points where the window fully overlaps the data
#     return np.convolve(data, kernel, mode='valid')

# # ---------------------------------------------------------
# # 2. Load Data
# # ---------------------------------------------------------
# try:
#     # Replace with your actual paths
#     baseline_loss = np.load('loss_baseline.npy')
#     main_loss = np.load('loss_hybrid.npy')
# except FileNotFoundError:
#     print("Files not found. Generating dummy data...")
#     x = np.linspace(0, 16000, 16000)
#     baseline_loss = 0.5 + 0.5 * np.exp(-x/4000) + np.random.normal(0, 0.05, 16000)
#     main_loss = 0.5 + 0.4 * np.exp(-x/4000) + np.random.normal(0, 0.05, 16000)

# # ---------------------------------------------------------
# # 3. Apply Smoothing
# # ---------------------------------------------------------
# # WINDOW SIZE: The larger this number, the smoother the line.
# # For 16,000 data points (as seen in your image), try 100 to 300.
# window = 100 

# baseline_smooth = moving_average(baseline_loss, window_size=window)
# main_smooth = moving_average(main_loss, window_size=window)

# # ---------------------------------------------------------
# # 4. Plotting
# # ---------------------------------------------------------
# plt.figure(figsize=(12, 7), dpi=100)

# # Create x-axis ranges. 
# # Since 'valid' convolution shrinks the array by (window - 1), we adjust x-axis.
# x_raw_len = len(baseline_loss)
# x_smooth_start = window // 2
# x_smooth_end = x_raw_len - (window // 2) + (1 if window % 2 != 0 else 0)
# # Note: This is a simple alignment. For massive datasets, simple plotting is fine.
# x_smooth_axis = np.arange(len(baseline_smooth)) + (window - 1) // 2

# # --- Plot Baseline ---
# # Raw data (Very transparent)
# plt.plot(baseline_loss, color='#1f77b4', alpha=0.15, label='_nolegend_') # Standard Blue
# # Smoothed data (Solid)
# plt.plot(x_smooth_axis, baseline_smooth, color='#1f77b4', label='Baseline (Smoothed)', linewidth=2)

# # --- Plot Main/Hybrid Method ---
# # Raw data (Very transparent)
# plt.plot(main_loss, color='#ff7f0e', alpha=0.15, label='_nolegend_') # Standard Orange
# # Smoothed data (Solid)
# plt.plot(x_smooth_axis, main_smooth, color='#ff7f0e', label='Hybrid/Main (Smoothed)', linewidth=2)

# # Styling
# plt.title('Training Loss Comparison', fontsize=14)
# plt.xlabel('Steps', fontsize=12)
# plt.ylabel('Loss', fontsize=12)
# plt.legend(fontsize=12, loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.7)

# # plt.savefig('smoothed_loss.png')
# plt.show()