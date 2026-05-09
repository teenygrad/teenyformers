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

//! Multi-head attention layer.
//!
//! Projects Q/K/V, transposes to `[BH, S, D_head]`, applies RoPE, then
//! Flash Attention 2 (causal or non-causal), merges heads, and projects out.
//!
//! Layout flow:
//! ```text
//! [S, D] --(Wq)--> [S, D] --TransposeHeads--> [H, S, D_head] --RoPE--> [H, S, D_head]
//!                                                                          ↓  FlashAttn2
//! [S, D] <--(Wo)-- [S, D] <--MergeHeads-- [H, S, D_head]
//! ```

use teeny_core::{graph::SymTensor, name_scope::name_scope};

use crate::kernels::{CausalFlashAttn2Op, FlashAttn2Op, MergeHeadsOp, RopeOp, TransposeHeadsOp};

use super::sym_linear;

/// Whether the attention is bidirectional (encoder) or causal (decoder self-attn).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    /// Bidirectional — used in encoder self-attention and cross-attention.
    Bidirectional,
    /// Causal — decoder self-attention with upper-triangular mask.
    Causal,
}

/// Multi-head attention layer (forward graph recording).
///
/// Records the following ops into the shared graph:
/// 1. Linear projections W_q, W_k, W_v (no bias)
/// 2. `TransposeHeads` — `[S, D]` → `[H, S, D_head]`
/// 3. `RoPE` — applied to Q and K
/// 4. Flash Attention 2 (causal or bidirectional)
/// 5. `MergeHeads` — `[H, S, D_head]` → `[S, D]`
/// 6. Output projection W_o (no bias)
pub struct MultiHeadAttention {
    pub d_model:   usize,
    pub n_heads:   usize,
    pub head_dim:  usize,
    pub rope_base: f32,
    pub kind:      AttentionKind,
}

impl MultiHeadAttention {
    pub fn new(
        d_model:   usize,
        n_heads:   usize,
        rope_base: f32,
        kind:      AttentionKind,
    ) -> Self {
        assert_eq!(d_model % n_heads, 0, "d_model must be divisible by n_heads");
        Self { d_model, n_heads, head_dim: d_model / n_heads, rope_base, kind }
    }

    /// Record self-attention: `q = x`, `k = x`, `v = x`.
    pub fn self_attn(&self, x: SymTensor) -> SymTensor {
        self.forward(x.clone(), x.clone(), x)
    }

    /// Record cross-attention: queries from `x`, keys/values from `context`.
    pub fn cross_attn(&self, x: SymTensor, context: SymTensor) -> SymTensor {
        self.forward(x, context.clone(), context)
    }

    /// Core forward: projects `q_in/k_in/v_in`, applies RoPE + attention.
    pub fn forward(
        &self,
        q_in: SymTensor,
        k_in: SymTensor,
        v_in: SymTensor,
    ) -> SymTensor {
        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        // Linear projections: [S, D] → [S, D]
        let q = { let _g = name_scope("q_proj"); sym_linear(q_in, self.d_model, self.d_model) };
        let k = { let _g = name_scope("k_proj"); sym_linear(k_in, self.d_model, self.d_model) };
        let v = { let _g = name_scope("v_proj"); sym_linear(v_in, self.d_model, self.d_model) };

        // Transpose heads: [S, D] → [H, S, D_head]
        let q = q.record_custom(
            TransposeHeadsOp::custom_data(self.n_heads, self.head_dim),
            &[],
            None,
        );
        let k = k.record_custom(
            TransposeHeadsOp::custom_data(self.n_heads, self.head_dim),
            &[],
            None,
        );
        let v = v.record_custom(
            TransposeHeadsOp::custom_data(self.n_heads, self.head_dim),
            &[],
            None,
        );

        // RoPE on Q and K
        let q = q.record_custom(
            RopeOp::custom_data(self.head_dim, self.rope_base),
            &[],
            None,
        );
        let k = k.record_custom(
            RopeOp::custom_data(self.head_dim, self.rope_base),
            &[],
            None,
        );

        // Flash Attention 2
        let o = match self.kind {
            AttentionKind::Bidirectional => q.record_custom(
                FlashAttn2Op::custom_data(self.head_dim, scale),
                &[&k, &v],
                None,
            ),
            AttentionKind::Causal => q.record_custom(
                CausalFlashAttn2Op::custom_data(self.head_dim, scale),
                &[&k, &v],
                None,
            ),
        };

        // Merge heads: [H, S, D_head] → [S, D]
        let o = o.record_custom(
            MergeHeadsOp::custom_data(self.n_heads, self.head_dim),
            &[],
            None,
        );

        // Output projection
        { let _g = name_scope("o_proj"); sym_linear(o, self.d_model, self.d_model) }
    }
}
