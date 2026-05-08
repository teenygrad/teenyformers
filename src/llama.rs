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

//! Llama-style decoder-only language model.
//!
//! Architecture (Llama 2 / Llama 3 naming):
//! ```text
//! ids [S]
//!   → TokenEmbedding          [S, D]
//!   → LlamaBlock × n_layers   [S, D]
//!   → final RMSNorm           [S, D]
//!   → lm_head (linear, no bias, weight-tied to embedding)  [S, vocab_size]
//!   → logits
//! ```
//!
//! Weight tying: the output projection (`lm_head`) shares weights with the
//! token embedding table.

use teeny_core::graph::SymTensor;

use crate::{
    config::LlamaConfig,
    layers::{LlamaBlock, TokenEmbedding, sym_rmsnorm},
};

/// Llama-style decoder-only transformer.
pub struct Llama {
    pub embedding: TokenEmbedding,
    pub layers:    Vec<LlamaBlock>,
    pub d_model:   usize,
    pub eps:       f64,
}

impl Llama {
    pub fn new(cfg: &LlamaConfig) -> Self {
        let d_ff = cfg.d_ff();
        Self {
            embedding: TokenEmbedding::new(cfg.vocab_size, cfg.d_model),
            layers: (0..cfg.n_layers)
                .map(|_| LlamaBlock::new(cfg.d_model, cfg.n_heads, d_ff, cfg.rope_base, cfg.eps))
                .collect(),
            d_model: cfg.d_model,
            eps: cfg.eps,
        }
    }

    /// Forward pass.
    ///
    /// - `ids [S]` — input token IDs (f32-encoded)
    ///
    /// Returns `logits [S, vocab_size]`.
    pub fn forward(&self, ids: SymTensor) -> SymTensor {
        let mut x = self.embedding.forward(ids);

        for block in &self.layers {
            x = block.forward(x);
        }

        let x = sym_rmsnorm(x, self.d_model, self.eps);
        self.embedding.output_proj(x)
    }
}
