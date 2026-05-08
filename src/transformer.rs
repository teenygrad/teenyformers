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

//! Full encoder-decoder transformer with weight-tied embeddings.
//!
//! ```text
//! src_ids → Embedding → Encoder → enc_out ─────────────────────────────────┐
//!                                                                           │
//! tgt_ids → Embedding → Decoder(enc_out) → final_norm → lm_head → logits  │
//!                               ↑───────────────────────────────────────────┘
//!                          cross-attn uses enc_out
//! ```
//!
//! Weight tying: the encoder embedding, decoder embedding, and output `lm_head`
//! projection all share the same `weight [vocab_size, d_model]` parameter.

use teeny_core::graph::SymTensor;

use crate::{
    config::TransformerConfig,
    decoder::Decoder,
    encoder::Encoder,
    layers::TokenEmbedding,
};

/// Full encoder-decoder transformer.
pub struct Transformer {
    pub embedding: TokenEmbedding,
    pub encoder:   Encoder,
    pub decoder:   Decoder,
}

impl Transformer {
    pub fn new(cfg: &TransformerConfig) -> Self {
        Self {
            embedding: TokenEmbedding::new(cfg.vocab_size, cfg.d_model),
            encoder:   Encoder::new(cfg),
            decoder:   Decoder::new(cfg),
        }
    }

    /// Forward pass.
    ///
    /// - `src_ids [S_src]` — encoder input token IDs (f32-encoded)
    /// - `tgt_ids [S_tgt]` — decoder input token IDs (f32-encoded, shifted right)
    ///
    /// Returns `logits [S_tgt, vocab_size]`.
    pub fn forward(&self, src_ids: SymTensor, tgt_ids: SymTensor) -> SymTensor {
        let src_emb = self.embedding.forward(src_ids);
        let enc_out = self.encoder.forward(src_emb);

        let tgt_emb = self.embedding.forward(tgt_ids);
        let dec_out = self.decoder.forward(tgt_emb, enc_out);

        self.embedding.output_proj(dec_out)
    }
}
