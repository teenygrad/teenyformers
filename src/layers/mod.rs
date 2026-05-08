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

pub mod attention;
pub mod decoder_block;
pub mod embedding;
pub mod encoder_block;
pub mod ffn;

pub use attention::{AttentionKind, MultiHeadAttention};
pub use decoder_block::DecoderBlock;
pub use embedding::TokenEmbedding;
pub use encoder_block::EncoderBlock;
pub use ffn::FeedForward;

use teeny_core::graph::{Op, Shape, SymTensor};

/// Record a two-tensor element-wise addition into the shared graph.
///
/// Both tensors must share the same graph, shape, and dtype.
pub fn sym_add(a: &SymTensor, b: &SymTensor) -> SymTensor {
    debug_assert!(
        std::rc::Rc::ptr_eq(&a.graph, &b.graph),
        "sym_add: both tensors must share the same graph"
    );
    let shape: Shape = a.shape.clone();
    let dtype = a.dtype;
    let node_id = a
        .graph
        .borrow_mut()
        .add_node(Op::Add, vec![a.node_id, b.node_id], dtype, shape.clone());
    SymTensor { node_id, graph: a.graph.clone(), dtype, shape }
}

/// Record a `Linear` projection (no bias) into the graph.
///
/// Uses the `Layer<SymTensor>` impl from `teeny_core` so we don't have to
/// call the crate-private `SymTensor::record` method directly.
pub fn sym_linear(x: SymTensor, in_features: usize, out_features: usize) -> SymTensor {
    use teeny_core::nn::{Layer, linear::Linear};
    let layer = Linear::<f32, SymTensor, SymTensor, 2>::new(in_features, out_features, false);
    Layer::call(&layer, x)
}

/// Record an RMSNorm operation into the graph.
pub fn sym_rmsnorm(x: SymTensor, dim: usize, eps: f64) -> SymTensor {
    use teeny_core::nn::{Layer, rmsnorm::RmsNorm};
    let layer = RmsNorm::<f32, SymTensor, SymTensor, 2>::new(vec![dim]).with_eps(eps);
    Layer::call(&layer, x)
}
