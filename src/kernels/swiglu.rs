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

//! SwiGLU fused activation kernel.
//!
//! Input layout: `z [M, 2·d_ff]` — first `d_ff` columns are the gate half,
//! last `d_ff` columns are the up-projection half (concatenated along dim-1).
//!
//! **Forward**: `out[m, n] = silu(z[m, n]) · z[m, n + d_ff]`
//!   where `silu(x) = x · σ(x)`.
//!
//! **Backward**:
//! ```text
//! σ     = sigmoid(gate)
//! silu  = gate · σ
//! d_silu/d_gate = σ · (1 + gate · (1 − σ))
//! dz_gate[m,n]        = grad[m,n] · up[m,n] · d_silu/d_gate
//! dz_up  [m,n+d_ff]   = grad[m,n] · silu(gate[m,n])
//! ```
//!
//! Grid: `[M, 1, 1]` — one CTA per row.

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
    types::{AddOffsets, Comparison, Tensor},
    *,
};

// ── Forward ───────────────────────────────────────────────────────────────────

/// SwiGLU forward.
///
/// Grid: `[M, 1, 1]`.
#[kernel]
pub fn swiglu_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    z_ptr:   T::Pointer<D>,   // [M, 2·d_ff]  gate || up
    out_ptr: T::Pointer<D>,   // [M, d_ff]
    _M:      i32,
    d_ff:    i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row       = T::program_id(Axis::X);
    let row_gate  = row * d_ff * 2;
    let row_up    = row_gate + d_ff;
    let row_out   = row * d_ff;

    let zeros = T::zeros::<D>(&[BLOCK_N]);

    let mut n_start: i32 = 0;
    while n_start < d_ff {
        let col      = T::arange(0, BLOCK_N) + n_start;
        let mask     = col.lt(d_ff);

        let gate = T::load(z_ptr.add_offsets(col + row_gate), Some(mask), Some(zeros), &[], None, None, None, false);
        let up   = T::load(z_ptr.add_offsets(col + row_up),   Some(mask), Some(zeros), &[], None, None, None, false);

        let sig  = T::sigmoid(gate);
        let silu = gate * sig;

        T::store(out_ptr.add_offsets(col + row_out), silu * up, Some(mask), &[], None, None);
        n_start += BLOCK_N;
    }
}

// ── Backward ─────────────────────────────────────────────────────────────────

/// SwiGLU backward.
///
/// `dz` has the same shape as `z` — `[M, 2·d_ff]`.
///
/// Grid: `[M, 1, 1]`.
#[cfg(feature = "training")]
#[kernel]
pub fn swiglu_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    grad_ptr: T::Pointer<D>,   // [M, d_ff]      upstream gradient
    z_ptr:    T::Pointer<D>,   // [M, 2·d_ff]    saved activation input
    dz_ptr:   T::Pointer<D>,   // [M, 2·d_ff]    output grad wrt z
    _M:       i32,
    d_ff:     i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row      = T::program_id(Axis::X);
    let row_gate = row * d_ff * 2;
    let row_up   = row_gate + d_ff;
    let row_grad = row * d_ff;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let one   = T::cast::<f32, D>(T::full::<f32>(&[BLOCK_N], 1.0_f32), None, false);

    let mut n_start: i32 = 0;
    while n_start < d_ff {
        let col  = T::arange(0, BLOCK_N) + n_start;
        let mask = col.lt(d_ff);

        let grad = T::load(grad_ptr.add_offsets(col + row_grad), Some(mask), Some(zeros), &[], None, None, None, false);
        let gate = T::load(z_ptr.add_offsets(col + row_gate),    Some(mask), Some(zeros), &[], None, None, None, false);
        let up   = T::load(z_ptr.add_offsets(col + row_up),      Some(mask), Some(zeros), &[], None, None, None, false);

        let sig      = T::sigmoid(gate);
        let silu     = gate * sig;
        // d(silu)/d(gate) = σ(1 + gate(1 − σ))
        let d_silu   = sig * (one + gate * (one - sig));

        T::store(dz_ptr.add_offsets(col + row_gate), grad * up * d_silu, Some(mask), &[], None, None);
        T::store(dz_ptr.add_offsets(col + row_up),   grad * silu,        Some(mask), &[], None, None);
        n_start += BLOCK_N;
    }
}

// ── RuntimeOp — forward ───────────────────────────────────────────────────────

impl RuntimeOp for SwigluForward<f32> {
    fn n_activation_inputs(&self) -> usize { 1 }  // z [M, 2·d_ff]

    fn param_shapes(
        &self,
        _input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        let m    = output_shape[0] as i32;
        let d_ff = output_shape[1] as i32;
        // Kernel: (z_ptr, out_ptr, M, d_ff)
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(m);
        visitor.visit_i32(d_ff);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        _params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let m    = output_shape[0] as i32;
        let d_ff = output_shape[1] as i32;
        // Kernel: (grad_ptr, z_ptr, dz_ptr, M, d_ff)
        visitor.visit_ptr(grad_output);      // grad [M, d_ff]
        visitor.visit_ptr(inputs[0].0);      // z    [M, 2·d_ff]  saved
        visitor.visit_ptr(grad_inputs[0]);   // dz   [M, 2·d_ff]
        visitor.visit_i32(m);
        visitor.visit_i32(d_ff);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1]
    }
}

// ── CustomOp — graph wrapper ──────────────────────────────────────────────────

/// Graph node for the SwiGLU fused activation.
///
/// Input:  `z [M, 2·d_ff]`  (gate || up halves concatenated)
/// Output: `out [M, d_ff]`
pub struct SwigluOp {
    pub block_n: i32,
}

impl SwigluOp {
    pub fn new() -> Self {
        Self { block_n: 1024 }
    }

    pub fn custom_data() -> CustomData {
        CustomData::new(Self::new())
    }
}

impl Default for SwigluOp {
    fn default() -> Self { Self::new() }
}

impl CustomOp for SwigluOp {
    fn name(&self) -> &str { "swiglu" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        let shape = input_shapes[0];
        let mut out = shape.to_vec();
        let last = out.len() - 1;
        out[last] = out[last].map(|n| n / 2);
        out
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let k = SwigluForward::<f32>::new(self.block_n);
        Some((
            k.name.to_string(),
            k.source.clone(),
            "entry_point".to_string(),
            Arc::new(k),
        ))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        SwigluBackward::<f32>::new(self.block_n).source.clone()
    }
}
