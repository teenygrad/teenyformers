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

//! Integration tests: graph construction and shape inference.
//!
//! These tests do NOT require a GPU.  They verify that the symbolic graph
//! is built correctly and that shape inference produces the expected output
//! shapes for every layer.

use teeny_core::graph::{DtypeRepr, Op, SymTensor};
use teenyformers::{
    TransformerConfig,
    layers::{
        AttentionKind, DecoderBlock, EncoderBlock, FeedForward, MultiHeadAttention,
        TokenEmbedding,
    },
};

/// Create a 2-D symbolic input `[S, D]` and its associated graph.
fn input_2d(s: usize, d: usize) -> SymTensor {
    let (x, _graph) = SymTensor::input(DtypeRepr::F32, vec![Some(s), Some(d)]);
    x
}

/// Create a 1-D symbolic input `[S]`.
fn input_1d(s: usize) -> SymTensor {
    let (x, _graph) = SymTensor::input(DtypeRepr::F32, vec![Some(s)]);
    x
}

/// Add a second symbolic input to the same graph as `peer`.
fn input_like(peer: &SymTensor, shape: Vec<Option<usize>>) -> SymTensor {
    let node_id = peer
        .graph
        .borrow_mut()
        .add_node(Op::Input, vec![], peer.dtype, shape.clone());
    SymTensor { node_id, graph: peer.graph.clone(), dtype: peer.dtype, shape }
}

// ── Config ────────────────────────────────────────────────────────────────────

#[test]
fn test_config_d_ff_rounding() {
    let cfg = TransformerConfig { d_model: 512, ..Default::default() };
    assert_eq!(cfg.d_ff() % 64, 0);

    let cfg2 = TransformerConfig { d_model: 128, ..Default::default() };
    assert_eq!(cfg2.d_ff(), 384); // 8/3 * 128 ≈ 341.3 → ceil to multiple of 64 = 384
}

#[test]
fn test_config_head_dim() {
    let cfg = TransformerConfig { d_model: 512, n_heads: 8, ..Default::default() };
    assert_eq!(cfg.head_dim(), 64);
}

// ── FeedForward ───────────────────────────────────────────────────────────────

#[test]
fn test_ffn_shape() {
    let (d_model, d_ff, seq) = (64, 128, 8);
    let ffn = FeedForward::new(d_model, d_ff);
    let out = ffn.forward(input_2d(seq, d_model));
    assert_eq!(out.shape, vec![Some(seq), Some(d_model)]);
}

// ── MultiHeadAttention ────────────────────────────────────────────────────────

#[test]
fn test_self_attention_shape() {
    let (d_model, seq) = (64, 8);
    let attn = MultiHeadAttention::new(d_model, 4, 10_000.0, AttentionKind::Bidirectional);
    let out = attn.self_attn(input_2d(seq, d_model));
    assert_eq!(out.shape, vec![Some(seq), Some(d_model)]);
}

#[test]
fn test_causal_attention_shape() {
    let (d_model, seq) = (64, 8);
    let attn = MultiHeadAttention::new(d_model, 4, 10_000.0, AttentionKind::Causal);
    let out = attn.self_attn(input_2d(seq, d_model));
    assert_eq!(out.shape, vec![Some(seq), Some(d_model)]);
}

#[test]
fn test_cross_attention_shape() {
    let (d_model, sq, sk) = (64, 5, 7);
    let q = input_2d(sq, d_model);
    let ctx = input_like(&q, vec![Some(sk), Some(d_model)]);
    let attn = MultiHeadAttention::new(d_model, 4, 10_000.0, AttentionKind::Bidirectional);
    let out = attn.cross_attn(q, ctx);
    assert_eq!(out.shape, vec![Some(sq), Some(d_model)]);
}

// ── EncoderBlock ─────────────────────────────────────────────────────────────

#[test]
fn test_encoder_block_shape() {
    let cfg = TransformerConfig { d_model: 64, n_heads: 4, ..Default::default() };
    let block = EncoderBlock::new(cfg.d_model, cfg.n_heads, cfg.d_ff(), cfg.rope_base, cfg.eps as f64);
    let out = block.forward(input_2d(8, cfg.d_model));
    assert_eq!(out.shape, vec![Some(8), Some(cfg.d_model)]);
}

// ── DecoderBlock ─────────────────────────────────────────────────────────────

#[test]
fn test_decoder_block_shape() {
    let cfg = TransformerConfig { d_model: 64, n_heads: 4, ..Default::default() };
    let tgt = input_2d(5, cfg.d_model);
    let enc = input_like(&tgt, vec![Some(7), Some(cfg.d_model)]);
    let block = DecoderBlock::new(cfg.d_model, cfg.n_heads, cfg.d_ff(), cfg.rope_base, cfg.eps as f64);
    let out = block.forward(tgt, enc);
    assert_eq!(out.shape, vec![Some(5), Some(cfg.d_model)]);
}

// ── Embedding ─────────────────────────────────────────────────────────────────

#[test]
fn test_token_embedding_shape() {
    let (vocab, d_model, seq) = (1000, 64, 8);
    let emb = TokenEmbedding::new(vocab, d_model);
    let out = emb.forward(input_1d(seq));
    assert_eq!(out.shape, vec![Some(seq), Some(d_model)]);
}

#[test]
fn test_output_proj_shape() {
    let (vocab, d_model, seq) = (1000, 64, 8);
    let emb = TokenEmbedding::new(vocab, d_model);
    let logits = emb.output_proj(input_2d(seq, d_model));
    assert_eq!(logits.shape, vec![Some(seq), Some(vocab)]);
}

// ── Full transformer ──────────────────────────────────────────────────────────

#[test]
fn test_full_transformer_shape() {
    let cfg = TransformerConfig {
        d_model: 64,
        n_heads: 4,
        n_encoder_layers: 2,
        n_decoder_layers: 2,
        vocab_size: 1000,
        ..Default::default()
    };
    let model = teenyformers::Transformer::new(&cfg);

    let src_ids = input_1d(7);
    let tgt_ids = input_like(&src_ids, vec![Some(5)]);

    let logits = model.forward(src_ids, tgt_ids);
    assert_eq!(logits.shape, vec![Some(5), Some(cfg.vocab_size)]);
}
