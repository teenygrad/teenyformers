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

//! Transformer decoder block (pre-norm).
//!
//! Three sub-layers:
//! 1. **Causal self-attention** — masked so each position can only attend to past tokens.
//! 2. **Cross-attention** — queries from the decoder, keys/values from the encoder output.
//! 3. **SwiGLU FFN**
//!
//! ```text
//!   tgt ─── RMSNorm ─── CausalSelfAttn ─── Add ─── RMSNorm ─── CrossAttn(enc) ─── Add ─── RMSNorm ─── FFN ─── Add ─── out
//!     └─────────────────────────────────────┘   └────────────────────────────────────┘   └──────────────────────────────┘
//! ```

use teeny_core::graph::SymTensor;

use super::{AttentionKind, FeedForward, MultiHeadAttention, sym_add, sym_rmsnorm};

/// One decoder block.
pub struct DecoderBlock {
    pub self_attn:  MultiHeadAttention,
    pub cross_attn: MultiHeadAttention,
    pub ffn:        FeedForward,
    pub d_model:    usize,
    pub eps:        f64,
}

impl DecoderBlock {
    pub fn new(
        d_model:   usize,
        n_heads:   usize,
        d_ff:      usize,
        rope_base: f32,
        eps:       f64,
    ) -> Self {
        Self {
            self_attn:  MultiHeadAttention::new(d_model, n_heads, rope_base, AttentionKind::Causal),
            cross_attn: MultiHeadAttention::new(d_model, n_heads, rope_base, AttentionKind::Bidirectional),
            ffn:        FeedForward::new(d_model, d_ff),
            d_model,
            eps,
        }
    }

    /// Forward: `tgt [S_t, D]`, `enc [S_e, D]` → `out [S_t, D]`.
    pub fn forward(&self, tgt: SymTensor, enc: SymTensor) -> SymTensor {
        // Causal self-attention
        let norm1    = sym_rmsnorm(tgt.clone(), self.d_model, self.eps);
        let self_out = self.self_attn.self_attn(norm1);
        let x2       = sym_add(&tgt, &self_out);

        // Cross-attention (queries from decoder, keys/values from encoder)
        let norm2      = sym_rmsnorm(x2.clone(), self.d_model, self.eps);
        let cross_out  = self.cross_attn.cross_attn(norm2, enc);
        let x3         = sym_add(&x2, &cross_out);

        // FFN
        let norm3   = sym_rmsnorm(x3.clone(), self.d_model, self.eps);
        let ffn_out = self.ffn.forward(norm3);
        sym_add(&x3, &ffn_out)
    }
}
