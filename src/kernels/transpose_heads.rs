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

//! Layout transposition between `[S, D]` and `[H, S, D_HEAD]` formats.
//!
//! Flash Attention 2 and RoPE kernels consume `[BH, N_CTX, HEAD_DIM]` layout
//! while the rest of the transformer uses `[S, D]` (where `D = H * HEAD_DIM`).
//!
//! - **TransposeHeadsOp**: `[S, D]` → `[H, S, HEAD_DIM]`
//! - **MergeHeadsOp**:     `[H, S, HEAD_DIM]` → `[S, D]`
//!
//! The backward of each op is the other op (they are mutual inverses).
//!
//! Grid: `(S, H, 1)` — one CTA per `(sequence-position, head)` pair.

use core::any::Any;
use std::sync::Arc;

use teeny_core::{
    device::program::ArgVisitor,
    graph::{CustomData, CustomOp, Shape},
    model::RuntimeOp,
};
use teeny_macros::kernel;
use teeny_triton::triton::{
    Axis,
    types::{AddOffsets, Tensor},
    *,
};

// ── Kernels ───────────────────────────────────────────────────────────────────

/// `[S, D]` → `[H, S, HEAD_DIM]`  (split heads to front).
///
/// Grid: `(seq_len, n_heads, 1)`.
#[kernel]
pub fn transpose_heads<T: Triton, const HEAD_DIM: i32>(
    src_ptr: T::Pointer<f32>,
    dst_ptr: T::Pointer<f32>,
    seq_len: i32,
    n_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_s  = T::program_id(Axis::X);  // sequence position
    let pid_h  = T::program_id(Axis::Y);  // head index
    let d      = T::arange(0, HEAD_DIM);  // [0..HEAD_DIM)
    let d_model = n_heads * HEAD_DIM;

    // src layout: [S, H, HEAD_DIM] interleaved — element [s, h, d] is at s*D + h*HEAD_DIM + d
    let src_offs = d + pid_s * d_model + pid_h * HEAD_DIM;
    // dst layout: [H, S, HEAD_DIM] — element [h, s, d] is at h*S*HEAD_DIM + s*HEAD_DIM + d
    let dst_offs = d + pid_h * seq_len * HEAD_DIM + pid_s * HEAD_DIM;

    let val = T::load(src_ptr.add_offsets(src_offs), None, None, &[], None, None, None, false);
    T::store(dst_ptr.add_offsets(dst_offs), val, None, &[], None, None);
}

/// `[H, S, HEAD_DIM]` → `[S, D]`  (merge heads back, inverse of `transpose_heads`).
///
/// Grid: `(seq_len, n_heads, 1)`.
#[kernel]
pub fn merge_heads<T: Triton, const HEAD_DIM: i32>(
    src_ptr: T::Pointer<f32>,
    dst_ptr: T::Pointer<f32>,
    seq_len: i32,
    n_heads: i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_s  = T::program_id(Axis::X);
    let pid_h  = T::program_id(Axis::Y);
    let d      = T::arange(0, HEAD_DIM);
    let d_model = n_heads * HEAD_DIM;

    let src_offs = d + pid_h * seq_len * HEAD_DIM + pid_s * HEAD_DIM;
    let dst_offs = d + pid_s * d_model + pid_h * HEAD_DIM;

    let val = T::load(src_ptr.add_offsets(src_offs), None, None, &[], None, None, None, false);
    T::store(dst_ptr.add_offsets(dst_offs), val, None, &[], None, None);
}

// ── TransposeHeadsOp ──────────────────────────────────────────────────────────

impl RuntimeOp for TransposeHeads {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],  // [H, S, HEAD_DIM]
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let seq_len = output_shape[1] as i32;
        let n_heads = output_shape[0] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(seq_len);
        visitor.visit_i32(n_heads);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output_shape = [H, S, HEAD_DIM]
        let seq_len = output_shape[1] as u32;
        let n_heads = output_shape[0] as u32;
        [seq_len, n_heads, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        _inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        // Backward of transpose = merge (same kernel in reverse direction)
        let seq_len = output_shape[1] as i32;
        let n_heads = output_shape[0] as i32;
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(seq_len);
        visitor.visit_i32(n_heads);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let seq_len = output_shape[1] as u32;
        let n_heads = output_shape[0] as u32;
        [seq_len, n_heads, 1]
    }
}

/// Graph node: `[S, D]` → `[H, S, HEAD_DIM]`.
pub struct TransposeHeadsOp {
    pub n_heads:  usize,
    pub head_dim: usize,
}

impl TransposeHeadsOp {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self { n_heads, head_dim }
    }

    pub fn custom_data(n_heads: usize, head_dim: usize) -> CustomData {
        CustomData::new(Self::new(n_heads, head_dim))
    }
}

impl CustomOp for TransposeHeadsOp {
    fn name(&self) -> &str { "transpose_heads" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [S?, D]  →  output: [H, S?, HEAD_DIM]
        let seq = input_shapes[0][0]; // S (may be None for dynamic)
        vec![Some(self.n_heads), seq, Some(self.head_dim)]
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let k = TransposeHeads::new(self.head_dim as i32);
        Some((k.name.to_string(), k.source.clone(), "entry_point".to_string(), Arc::new(k)))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        MergeHeads::new(self.head_dim as i32).source.clone()
    }
}

// ── MergeHeadsOp ─────────────────────────────────────────────────────────────

impl RuntimeOp for MergeHeads {
    fn n_activation_inputs(&self) -> usize { 1 }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> { Vec::new() }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],  // [S, D]
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        // input is [H, S, HEAD_DIM]
        let seq_len = output_shape[0] as i32;
        let n_heads = (output_shape[1] / self.head_dim as usize) as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(seq_len);
        visitor.visit_i32(n_heads);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // output = [S, D]
        let seq_len = output_shape[0] as u32;
        let n_heads = (output_shape[1] / self.head_dim as usize) as u32;
        [seq_len, n_heads, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        _inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        // Backward of merge = transpose
        let seq_len = output_shape[0] as i32;
        let n_heads = (output_shape[1] / self.head_dim as usize) as i32;
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_i32(seq_len);
        visitor.visit_i32(n_heads);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let seq_len = output_shape[0] as u32;
        let n_heads = (output_shape[1] / self.head_dim as usize) as u32;
        [seq_len, n_heads, 1]
    }
}

/// Graph node: `[H, S, HEAD_DIM]` → `[S, D]`.
pub struct MergeHeadsOp {
    pub n_heads:  usize,
    pub head_dim: usize,
}

impl MergeHeadsOp {
    pub fn new(n_heads: usize, head_dim: usize) -> Self {
        Self { n_heads, head_dim }
    }

    pub fn custom_data(n_heads: usize, head_dim: usize) -> CustomData {
        CustomData::new(Self::new(n_heads, head_dim))
    }
}

impl CustomOp for MergeHeadsOp {
    fn name(&self) -> &str { "merge_heads" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        // input: [H, S?, HEAD_DIM]  →  output: [S?, D]
        let seq = input_shapes[0][1]; // S (may be None)
        let d   = Some(self.n_heads * self.head_dim);
        vec![seq, d]
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let k = MergeHeads::new(self.head_dim as i32);
        Some((k.name.to_string(), k.source.clone(), "entry_point".to_string(), Arc::new(k)))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        TransposeHeads::new(self.head_dim as i32).source.clone()
    }
}
