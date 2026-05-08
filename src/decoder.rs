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

//! Transformer decoder stack.
//!
//! N identical `DecoderBlock`s followed by a final `RMSNorm`.

use teeny_core::graph::SymTensor;

use crate::{config::TransformerConfig, layers::{DecoderBlock, sym_rmsnorm}};

/// Stacked decoder blocks + final norm.
pub struct Decoder {
    pub blocks:  Vec<DecoderBlock>,
    pub d_model: usize,
    pub eps:     f64,
}

impl Decoder {
    pub fn new(cfg: &TransformerConfig) -> Self {
        let eps = cfg.eps as f64;
        let blocks = (0..cfg.n_decoder_layers)
            .map(|_| DecoderBlock::new(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_ff(),
                cfg.rope_base,
                eps,
            ))
            .collect();
        Self { blocks, d_model: cfg.d_model, eps }
    }

    /// Forward: `tgt [S_t, D]`, `enc [S_e, D]` → `dec_out [S_t, D]`.
    pub fn forward(&self, mut tgt: SymTensor, enc: SymTensor) -> SymTensor {
        for block in &self.blocks {
            tgt = block.forward(tgt, enc.clone());
        }
        sym_rmsnorm(tgt, self.d_model, self.eps)
    }
}
