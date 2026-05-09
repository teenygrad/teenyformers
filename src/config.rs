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

/// Configuration for a decoder-only (Llama-style) transformer.
///
/// `d_ff` is either taken from `intermediate_size` (if set) or derived:
/// `ceil(8/3 · d_model / 64) · 64`.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Token embedding / hidden dimension.
    pub d_model: usize,
    /// Number of attention heads.  Must divide `d_model`.
    pub n_heads: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length supported by RoPE.
    pub max_seq_len: usize,
    /// RMSNorm stability constant (default 1e-5).
    pub eps: f64,
    /// RoPE base frequency (default 10 000.0 — original LLaMA/GPT-NeoX).
    pub rope_base: f32,
    /// Explicit FFN intermediate width.  When `None`, uses the LLaMA formula.
    pub intermediate_size: Option<usize>,
    /// Number of KV heads (for grouped-query attention).  `None` = same as `n_heads`.
    pub n_kv_heads: Option<usize>,
}

#[cfg(all(feature = "serde", feature = "serde_json"))]
mod hf {
    use serde::Deserialize;
    use super::LlamaConfig;

    /// Subset of HuggingFace `config.json` fields needed to reconstruct a `LlamaConfig`.
    #[derive(Debug, Deserialize)]
    pub struct LlamaHFConfig {
        pub hidden_size:            usize,
        pub num_attention_heads:    usize,
        pub num_hidden_layers:      usize,
        pub vocab_size:             usize,
        #[serde(default = "default_max_position_embeddings")]
        pub max_position_embeddings: usize,
        #[serde(default = "default_rms_norm_eps")]
        pub rms_norm_eps:           f64,
        #[serde(default = "default_rope_theta")]
        pub rope_theta:             f64,
        pub intermediate_size:      Option<usize>,
        pub num_key_value_heads:    Option<usize>,
    }

    fn default_max_position_embeddings() -> usize { 4096 }
    fn default_rms_norm_eps() -> f64 { 1e-5 }
    fn default_rope_theta() -> f64 { 10_000.0 }

    impl From<LlamaHFConfig> for LlamaConfig {
        fn from(hf: LlamaHFConfig) -> Self {
            LlamaConfig {
                d_model:           hf.hidden_size,
                n_heads:           hf.num_attention_heads,
                n_layers:          hf.num_hidden_layers,
                vocab_size:        hf.vocab_size,
                max_seq_len:       hf.max_position_embeddings,
                eps:               hf.rms_norm_eps,
                rope_base:         hf.rope_theta as f32,
                intermediate_size: hf.intermediate_size,
                n_kv_heads:        hf.num_key_value_heads,
            }
        }
    }

    impl LlamaConfig {
        /// Parse a HuggingFace `config.json` file into a `LlamaConfig`.
        pub fn from_hf_json(path: &std::path::Path) -> anyhow::Result<Self> {
            let text = std::fs::read_to_string(path)?;
            let hf: LlamaHFConfig = serde_json::from_str(&text)?;
            Ok(hf.into())
        }
    }
}

impl LlamaConfig {
    /// Head dimension (`d_model / n_heads`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// SwiGLU intermediate width.
    ///
    /// Uses `intermediate_size` if explicitly set; otherwise computes
    /// `ceil(8/3 · d_model / 64) · 64` (the original LLaMA formula).
    pub fn d_ff(&self) -> usize {
        if let Some(sz) = self.intermediate_size {
            return sz;
        }
        let raw = (8 * self.d_model).div_ceil(3);
        raw.div_ceil(64) * 64
    }

    /// Attention softmax scale `1 / √head_dim`.
    pub fn softmax_scale(&self) -> f32 {
        1.0 / (self.head_dim() as f32).sqrt()
    }

    /// Number of KV heads (defaults to `n_heads` for standard MHA).
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads.unwrap_or(self.n_heads)
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            d_model:           4096,
            n_heads:           32,
            n_layers:          32,
            vocab_size:        32_000,
            max_seq_len:       4096,
            eps:               1e-5,
            rope_base:         10_000.0,
            intermediate_size: None,
            n_kv_heads:        None,
        }
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
