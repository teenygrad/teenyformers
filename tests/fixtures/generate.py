#!/usr/bin/env python3
"""Generate binary test fixtures for teenyformers integration tests.

Each fixture is a directory containing:
  input*.bin  — raw little-endian f32 arrays
  output.bin  — expected raw f32 output
  meta.json   — shape / config metadata

Run with: python3 tests/fixtures/generate.py
Requires: torch >= 2.0
"""

import json
import os
import struct
import sys

import torch
import torch.nn.functional as F

FIXTURES_DIR = os.path.dirname(__file__)
SEED = 42
torch.manual_seed(SEED)


def save_f32(path: str, t: torch.Tensor) -> None:
    arr = t.detach().float().cpu().contiguous()
    with open(path, "wb") as f:
        f.write(struct.pack(f"{arr.numel()}f", *arr.flatten().tolist()))


def save_meta(directory: str, meta: dict) -> None:
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def rand(*shape) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)


# ── RoPE ─────────────────────────────────────────────────────────────────────

def rope_rotate(x: torch.Tensor, rope_base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to x [..., HEAD_DIM]. Interleaved pair rotation."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    # positions = the sequence dim — for simplicity use first dim
    # x shape: [BH, S, HEAD_DIM]
    bh, s, d = x.shape
    k = torch.arange(half, dtype=torch.float32)  # [half]
    inv_freq = rope_base ** (-2 * k / d)         # [half]
    # pos: [S]
    pos = torch.arange(s, dtype=torch.float32)
    # theta: [S, half]
    theta = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_t = torch.cos(theta)  # [S, half]
    sin_t = torch.sin(theta)

    # Pair-interleaved: y[2k] = x[2k]*cos - x[2k+1]*sin
    x_even = x[..., 0::2]  # [BH, S, half]
    x_odd  = x[..., 1::2]
    y_even = x_even * cos_t - x_odd * sin_t
    y_odd  = x_even * sin_t + x_odd * cos_t

    y = torch.stack([y_even, y_odd], dim=-1).flatten(-2)  # [BH, S, HEAD_DIM]
    return y


def gen_rope():
    d = os.path.join(FIXTURES_DIR, "rope")
    bh, s, head_dim = 2, 8, 16
    rope_base = 10000.0
    x = rand(bh, s, head_dim)
    y = rope_rotate(x, rope_base)

    save_f32(os.path.join(d, "input.bin"), x)
    save_f32(os.path.join(d, "output.bin"), y)
    save_meta(d, {"bh": bh, "s": s, "head_dim": head_dim, "rope_base": rope_base})
    print(f"[rope] x{list(x.shape)} → y{list(y.shape)}")


# ── SwiGLU ───────────────────────────────────────────────────────────────────

def swiglu(z: torch.Tensor) -> torch.Tensor:
    """z [M, 2*d_ff] → out [M, d_ff]: silu(gate) * up."""
    half = z.shape[-1] // 2
    gate = z[..., :half]
    up   = z[..., half:]
    return F.silu(gate) * up


def gen_swiglu():
    d = os.path.join(FIXTURES_DIR, "swiglu")
    m, d_ff = 4, 8
    z = rand(m, 2 * d_ff)
    out = swiglu(z)

    save_f32(os.path.join(d, "input.bin"), z)
    save_f32(os.path.join(d, "output.bin"), out)
    save_meta(d, {"m": m, "d_ff": d_ff})
    print(f"[swiglu] z{list(z.shape)} → out{list(out.shape)}")


# ── FusedAddRmsNorm ───────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return x * rms * weight


def gen_fused_add_rmsnorm():
    d = os.path.join(FIXTURES_DIR, "fused_add_rmsnorm")
    m, n = 4, 16
    eps = 1e-6
    x        = rand(m, n)
    residual = rand(m, n)
    weight   = rand(n)

    h = x + residual
    y = rms_norm(h, weight, eps)

    save_f32(os.path.join(d, "input_x.bin"),        x)
    save_f32(os.path.join(d, "input_residual.bin"),  residual)
    save_f32(os.path.join(d, "weight.bin"),          weight)
    save_f32(os.path.join(d, "output_y.bin"),        y)
    save_f32(os.path.join(d, "output_h.bin"),        h)
    save_meta(d, {"m": m, "n": n, "eps": eps})
    print(f"[fused_add_rmsnorm] x{list(x.shape)} → y{list(y.shape)}")


# ── EncoderBlock ─────────────────────────────────────────────────────────────

class EncoderBlock(torch.nn.Module):
    """Pre-norm encoder block: RMSNorm + MHA + residual + RMSNorm + FFN + residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = torch.nn.RMSNorm(d_model, eps=eps)
        self.norm2 = torch.nn.RMSNorm(d_model, eps=eps)

        self.wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.wo = torch.nn.Linear(d_model, d_model, bias=False)

        self.wgate_up = torch.nn.Linear(d_model, 2 * d_ff, bias=False)
        self.wdown    = torch.nn.Linear(d_ff, d_model, bias=False)

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h, dh = self.n_heads, self.head_dim
        q = self.wq(x).view(b, s, h, dh).transpose(1, 2)
        k = self.wk(x).view(b, s, h, dh).transpose(1, 2)
        v = self.wv(x).view(b, s, h, dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.wo(out)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        z = self.wgate_up(x)
        return self.wdown(swiglu(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.norm1(x))
        x = x + self._ffn(self.norm2(x))
        return x


def gen_encoder_block():
    d = os.path.join(FIXTURES_DIR, "encoder_block")
    d_model, n_heads, d_ff, seq = 32, 4, 48, 5
    block = EncoderBlock(d_model, n_heads, d_ff)
    block.eval()

    x = rand(1, seq, d_model)  # [B=1, S, D]
    with torch.no_grad():
        y = block(x)

    save_f32(os.path.join(d, "input.bin"),  x.squeeze(0))
    save_f32(os.path.join(d, "output.bin"), y.squeeze(0))
    meta = {
        "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff, "seq": seq,
        "weights": {}
    }
    # Save all weights
    for name, param in block.named_parameters():
        fname = name.replace(".", "_") + ".bin"
        save_f32(os.path.join(d, fname), param)
        meta["weights"][name] = list(param.shape)
    save_meta(d, meta)
    print(f"[encoder_block] x{list(x.squeeze(0).shape)} → y{list(y.squeeze(0).shape)}")


# ── DecoderBlock ─────────────────────────────────────────────────────────────

class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = torch.nn.RMSNorm(d_model, eps=eps)
        self.norm2 = torch.nn.RMSNorm(d_model, eps=eps)
        self.norm3 = torch.nn.RMSNorm(d_model, eps=eps)

        # Causal self-attention
        self.wqs = torch.nn.Linear(d_model, d_model, bias=False)
        self.wks = torch.nn.Linear(d_model, d_model, bias=False)
        self.wvs = torch.nn.Linear(d_model, d_model, bias=False)
        self.wos = torch.nn.Linear(d_model, d_model, bias=False)

        # Cross-attention
        self.wqc = torch.nn.Linear(d_model, d_model, bias=False)
        self.wkc = torch.nn.Linear(d_model, d_model, bias=False)
        self.wvc = torch.nn.Linear(d_model, d_model, bias=False)
        self.woc = torch.nn.Linear(d_model, d_model, bias=False)

        self.wgate_up = torch.nn.Linear(d_model, 2 * d_ff, bias=False)
        self.wdown    = torch.nn.Linear(d_ff, d_model, bias=False)

    def _attn(self, q_in, kv_in, causal=False):
        b, sq, d = q_in.shape
        sk = kv_in.shape[1]
        h, dh = self.n_heads, self.head_dim
        q = q_in
        k = kv_in
        v = kv_in

        if causal:
            wq, wk, wv, wo = self.wqs, self.wks, self.wvs, self.wos
        else:
            wq, wk, wv, wo = self.wqc, self.wkc, self.wvc, self.woc

        q = wq(q).view(b, sq, h, dh).transpose(1, 2)
        k = wk(k).view(b, sk, h, dh).transpose(1, 2)
        v = wv(v).view(b, sk, h, dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).contiguous().view(b, sq, d)
        return wo(out)

    def _ffn(self, x):
        return self.wdown(swiglu(self.wgate_up(x)))

    def forward(self, tgt, enc):
        tgt = tgt + self._attn(self.norm1(tgt), self.norm1(tgt), causal=True)
        tgt = tgt + self._attn(self.norm2(tgt), enc, causal=False)
        tgt = tgt + self._ffn(self.norm3(tgt))
        return tgt


def gen_decoder_block():
    d = os.path.join(FIXTURES_DIR, "decoder_block")
    d_model, n_heads, d_ff, st, se = 32, 4, 48, 5, 7
    block = DecoderBlock(d_model, n_heads, d_ff)
    block.eval()

    tgt = rand(1, st, d_model)
    enc = rand(1, se, d_model)
    with torch.no_grad():
        out = block(tgt, enc)

    save_f32(os.path.join(d, "input_tgt.bin"), tgt.squeeze(0))
    save_f32(os.path.join(d, "input_enc.bin"), enc.squeeze(0))
    save_f32(os.path.join(d, "output.bin"),    out.squeeze(0))
    meta = {
        "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
        "seq_tgt": st, "seq_enc": se, "weights": {},
    }
    for name, param in block.named_parameters():
        fname = name.replace(".", "_") + ".bin"
        save_f32(os.path.join(d, fname), param)
        meta["weights"][name] = list(param.shape)
    save_meta(d, meta)
    print(f"[decoder_block] tgt{list(tgt.squeeze(0).shape)}, enc{list(enc.squeeze(0).shape)} → out{list(out.squeeze(0).shape)}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen_rope()
    gen_swiglu()
    gen_fused_add_rmsnorm()
    gen_encoder_block()
    gen_decoder_block()
    print("All fixtures generated.")
