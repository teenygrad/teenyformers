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

//! Token embedding layer with weight tying.
//!
//! The embedding table `E [vocab_size, d_model]` is shared between:
//! - Input token lookup (encoder and decoder)
//! - Output logit projection (`E^T`)
//!
//! Graph representation: a single `Op::Custom(TokenEmbedOp)` records the
//! embedding lookup.  The tied output projection is a separate linear layer
//! that *shares the same weight slot index* (`params[0]`).
//!
//! In both cases `param_names()` returns `["weight"]`, so the executor loads
//! the same file into both ops' `params[0]`.

use core::any::Any;
use std::sync::Arc;

use teeny_core::{
    device::program::ArgVisitor,
    graph::{CustomData, CustomOp, DtypeRepr, Shape, SymTensor},
    model::RuntimeOp,
};

// ── RuntimeOp ─────────────────────────────────────────────────────────────────

/// RuntimeOp: token lookup `ids [S] → embeddings [S, D]`.
///
/// Implemented as a row-gather (one row per token ID), equivalent to
/// `E[ids, :]` where `E [vocab_size, D]` is `params[0]`.
pub struct TokenEmbedRuntimeOp {
    pub vocab_size: usize,
    pub d_model:    usize,
}

impl RuntimeOp for TokenEmbedRuntimeOp {
    fn n_activation_inputs(&self) -> usize { 1 } // ids [S]

    fn param_shapes(
        &self,
        _input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> Vec<Vec<usize>> {
        vec![vec![self.vocab_size, self.d_model]] // params[0] = embedding table
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"] // loadable from checkpoint
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let seq_len = output_shape[0] as i32;
        let d_model  = output_shape[1] as i32;
        visitor.visit_ptr(inputs[0].0); // ids_ptr  [S]
        visitor.visit_ptr(params[0]);    // weight_ptr [vocab, D]
        visitor.visit_ptr(output);       // out_ptr  [S, D]
        visitor.visit_i32(seq_len);
        visitor.visit_i32(d_model);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1] // one CTA per token
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { false } // embedding backward is scatter-add; not needed for forward-only
}

// ── CustomOp ─────────────────────────────────────────────────────────────────

/// Graph node for the token embedding lookup.
///
/// Input:  `ids [S]` (integer token IDs; stored as f32 in the graph)
/// Output: `x [S, D]`
pub struct TokenEmbedOp {
    pub vocab_size: usize,
    pub d_model:    usize,
}

impl TokenEmbedOp {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self { vocab_size, d_model }
    }

    pub fn custom_data(vocab_size: usize, d_model: usize) -> CustomData {
        CustomData::new(Self::new(vocab_size, d_model))
    }
}

impl CustomOp for TokenEmbedOp {
    fn name(&self) -> &str { "token_embed" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let seq = input_shapes[0][0]; // S (may be None)
        vec![seq, Some(self.d_model)]
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let rop = TokenEmbedRuntimeOp { vocab_size: self.vocab_size, d_model: self.d_model };
        // Reuse the linear forward source for the entry-point wrapper; the
        // executor calls our RuntimeOp::pack_args, not the inner kernel.
        let src = String::new(); // executor uses RuntimeOp directly
        Some(("token_embed".to_string(), src, "entry_point".to_string(), Arc::new(rop)))
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Token embedding layer (forward graph recording only).
pub struct TokenEmbedding {
    pub vocab_size: usize,
    pub d_model:    usize,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, d_model: usize) -> Self {
        Self { vocab_size, d_model }
    }

    /// Record an embedding lookup for `ids [S]` → `[S, D]`.
    pub fn forward(&self, ids: SymTensor) -> SymTensor {
        ids.record_custom(
            TokenEmbedOp::custom_data(self.vocab_size, self.d_model),
            &[],
            Some(DtypeRepr::F32),
        )
    }

    /// Record a tied output projection `x [S, D]` → logits `[S, vocab_size]`.
    ///
    /// This is `x @ E^T` where `E` is the same weight as the embedding table.
    /// The output projection is recorded as `Op::Linear` so it shares the same
    /// weight-name convention with the embedding lookup.
    pub fn output_proj(&self, x: SymTensor) -> SymTensor {
        use teeny_core::nn::{Layer, linear::Linear};
        let layer = Linear::<f32, SymTensor, SymTensor, 2>::new(
            self.d_model,
            self.vocab_size,
            false,
        );
        Layer::call(&layer, x)
    }
}
