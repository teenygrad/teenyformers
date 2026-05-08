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

/// Configuration for the full encoder-decoder transformer.
///
/// `d_ff` is derived rather than stored: `d_ff = ceil(8/3 * d_model / 64) * 64`.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Token embedding / hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.  Must divide `d_model`.
    pub n_heads: usize,
    /// Encoder stack depth.
    pub n_encoder_layers: usize,
    /// Decoder stack depth.
    pub n_decoder_layers: usize,
    /// Vocabulary size (shared encoder/decoder/output embedding).
    pub vocab_size: usize,
    /// Maximum sequence length supported by RoPE precomputation.
    pub max_seq_len: usize,
    /// RMSNorm stability constant (default 1e-6).
    pub eps: f32,
    /// RoPE base frequency (default 10 000.0 — matches LLaMA / GPT-NeoX).
    pub rope_base: f32,
}

impl TransformerConfig {
    /// Head dimension (`d_model / n_heads`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// SwiGLU intermediate width: `ceil(8/3 * d_model / 64) * 64`.
    pub fn d_ff(&self) -> usize {
        let raw = (8 * self.d_model).div_ceil(3);
        raw.div_ceil(64) * 64
    }

    /// Attention softmax scale `1 / √head_dim`.
    pub fn softmax_scale(&self) -> f32 {
        1.0 / (self.head_dim() as f32).sqrt()
    }
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            n_encoder_layers: 6,
            n_decoder_layers: 6,
            vocab_size: 32_000,
            max_seq_len: 2048,
            eps: 1e-6,
            rope_base: 10_000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_ff_multiples_of_64() {
        let cfg = TransformerConfig { d_model: 128, ..Default::default() };
        assert_eq!(cfg.d_ff() % 64, 0);
        // 8/3 * 128 = 341.3 → ceil to nearest 64 multiple = 384
        assert_eq!(cfg.d_ff(), 384);
    }

    #[test]
    fn test_head_dim() {
        let cfg = TransformerConfig { d_model: 512, n_heads: 8, ..Default::default() };
        assert_eq!(cfg.head_dim(), 64);
    }
}
