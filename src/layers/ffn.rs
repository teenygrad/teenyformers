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

//! SwiGLU feed-forward network layer.
//!
//! Architecture (LLaMA / modern transformer):
//! ```text
//! x [S, D] --W_gate+W_up (concat)--> [S, 2*d_ff] --SwiGLU--> [S, d_ff] --W_down--> [S, D]
//! ```
//!
//! The up and gate projections are fused into one `[D, 2*d_ff]` weight matrix
//! so the SwiGLU kernel receives its expected `[S, 2*d_ff]` input in one load.
//!
//! `d_ff = ceil(8/3 * d_model / 64) * 64`  (LLaMA rounding)

use teeny_core::{graph::SymTensor, name_scope::name_scope};

use crate::kernels::SwigluOp;

use super::sym_linear;

/// SwiGLU feed-forward network.
pub struct FeedForward {
    pub d_model: usize,
    pub d_ff:    usize,
}

impl FeedForward {
    pub fn new(d_model: usize, d_ff: usize) -> Self {
        Self { d_model, d_ff }
    }

    /// Record the FFN computation: `[S, D] → [S, D]`.
    pub fn forward(&self, x: SymTensor) -> SymTensor {
        // Fused gate+up projection: [S, D] → [S, 2*d_ff]
        let z = { let _g = name_scope("gate_up_proj"); sym_linear(x, self.d_model, 2 * self.d_ff) };

        // SwiGLU: [S, 2*d_ff] → [S, d_ff]
        let h = z.record_custom(SwigluOp::custom_data(), &[], None);

        // Down projection: [S, d_ff] → [S, D]
        { let _g = name_scope("down_proj"); sym_linear(h, self.d_ff, self.d_model) }
    }
}
