/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Llama-style decoder-only transformer block (pre-norm).
//!
//! ```text
//!   x ─── RMSNorm ─── CausalSelfAttn ─── Add ─── RMSNorm ─── SwiGLU-FFN ─── Add ─── out
//!     └────────────────────────────────────┘   └──────────────────────────────────────┘
//! ```
//!
//! No cross-attention; causal masking only.

use teeny_core::graph::SymTensor;

use super::{AttentionKind, FeedForward, MultiHeadAttention, sym_add, sym_rmsnorm};

/// One Llama-style decoder-only transformer block.
pub struct LlamaBlock {
    pub self_attn: MultiHeadAttention,
    pub ffn:       FeedForward,
    pub d_model:   usize,
    pub eps:       f64,
}

impl LlamaBlock {
    pub fn new(
        d_model:   usize,
        n_heads:   usize,
        d_ff:      usize,
        rope_base: f32,
        eps:       f64,
    ) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, n_heads, rope_base, AttentionKind::Causal),
            ffn:       FeedForward::new(d_model, d_ff),
            d_model,
            eps,
        }
    }

    /// Forward: `x [S, D]` → `out [S, D]`.
    pub fn forward(&self, x: SymTensor) -> SymTensor {
        // Pre-norm causal self-attention + residual
        let norm1 = sym_rmsnorm(x.clone(), self.d_model, self.eps);
        let attn  = self.self_attn.self_attn(norm1);
        let x2    = sym_add(&x, &attn);

        // Pre-norm SwiGLU FFN + residual
        let norm2   = sym_rmsnorm(x2.clone(), self.d_model, self.eps);
        let ffn_out = self.ffn.forward(norm2);
        sym_add(&x2, &ffn_out)
    }
}
