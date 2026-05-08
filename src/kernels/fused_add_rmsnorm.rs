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

//! Fused residual-add + RMSNorm kernels.
//!
//! Computes `h = x + residual` and then RMS-normalises `h` in a single CTA
//! pass, halving memory traffic compared to separate add + normalise ops.
//!
//! Layout: `x` and `residual` are `[M, N]` row-major.  One CTA per row.
//!
//! **Forward outputs**
//! - `y_ptr`   — normalised result `h / rms(h) * γ`   (`[M, N]`)
//! - `h_ptr`   — updated residual `x + residual`       (`[M, N]`)  saved for backward
//! - `rstd_ptr`— reciprocal std `1 / rms(h)`           (`[M]`)     saved for backward
//!
//! **Backward** reconstructs `h` from `inputs` (x + residual) and uses saved `rstd`.
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

/// Fused residual-add + RMSNorm forward.
///
/// Grid: `[M, 1, 1]`.
#[kernel]
pub fn fused_add_rmsnorm_forward<T: Triton, D: Float, const BLOCK_N: i32>(
    x_ptr:       T::Pointer<D>,
    residual_ptr: T::Pointer<D>,
    y_ptr:       T::Pointer<D>,
    h_ptr:       T::Pointer<D>,
    rstd_ptr:    T::Pointer<D>,
    weight_ptr:  T::Pointer<D>,
    _M:          i32,
    N:           i32,
    eps:         f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row       = T::program_id(Axis::X);
    let row_start = row * N;
    let row_idx   = T::arange(0, 1) + row;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero1 = T::zeros::<D>(&[1]);
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0_f32 / (N as f32)), None, false);

    // Pass 1: compute h = x + residual; accumulate Σ h²
    let mut sq_sum = zero1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col = T::arange(0, BLOCK_N) + n_start;
        let mask = col.lt(N);

        let xi = T::load(x_ptr.add_offsets(col + row_start),        Some(mask), Some(zeros), &[], None, None, None, false);
        let ri = T::load(residual_ptr.add_offsets(col + row_start),  Some(mask), Some(zeros), &[], None, None, None, false);
        let hi = xi + ri;

        T::store(h_ptr.add_offsets(col + row_start), hi, Some(mask), &[], None, None);
        sq_sum = sq_sum + T::sum(hi * hi, None, true);
        n_start += BLOCK_N;
    }

    let eps_t  = T::cast::<f32, D>(T::full::<f32>(&[1], eps), None, false);
    let rstd_1 = T::rsqrt(sq_sum * n_inv + eps_t);
    T::store(rstd_ptr.add_offsets(row_idx), rstd_1, None, &[], None, None);

    let rstd = T::broadcast_to(rstd_1, &[BLOCK_N]);

    // Pass 2: y = h * rstd * γ
    n_start = 0;
    while n_start < N {
        let col  = T::arange(0, BLOCK_N) + n_start;
        let mask = col.lt(N);

        let hi = T::load(h_ptr.add_offsets(col + row_start),    Some(mask), Some(zeros), &[], None, None, None, false);
        let wi = T::load(weight_ptr.add_offsets(col),            Some(mask), Some(zeros), &[], None, None, None, false);

        T::store(y_ptr.add_offsets(col + row_start), hi * rstd * wi, Some(mask), &[], None, None);
        n_start += BLOCK_N;
    }
}

// ── Backward ─────────────────────────────────────────────────────────────────

/// Fused residual-add + RMSNorm backward.
///
/// Reads `h = x + residual` directly from the saved `h_ptr` buffer.
///
/// Gradient equations (same as plain RMSNorm backward, but `d_x = d_residual`
/// since both flow through the same addition):
/// ```text
/// dot   = (1/N) Σ_n dy_n · γ_n · h_n
/// dx[n] = rstd · γ[n] · (dy[n] − h[n] · rstd² · dot)
/// dγ[n] = Σ_m dy[m,n] · h[m,n] · rstd[m]
/// ```
///
/// Grid: `[M, 1, 1]`.
#[cfg(feature = "training")]
#[kernel]
pub fn fused_add_rmsnorm_backward<T: Triton, D: Float, const BLOCK_N: i32>(
    dy_ptr:     T::Pointer<D>,
    h_ptr:      T::Pointer<D>,
    dx_ptr:     T::Pointer<D>,   // also written to dresidual_ptr (same value)
    dresidual_ptr: T::Pointer<D>,
    dweight_ptr: T::Pointer<D>,
    weight_ptr:  T::Pointer<D>,
    rstd_ptr:    T::Pointer<D>,
    _M:          i32,
    N:           i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let row       = T::program_id(Axis::X);
    let row_start = row * N;
    let row_idx   = T::arange(0, 1) + row;

    let zeros = T::zeros::<D>(&[BLOCK_N]);
    let zero1 = T::zeros::<D>(&[1]);
    let n_inv = T::cast::<f32, D>(T::full::<f32>(&[1], 1.0_f32 / (N as f32)), None, false);

    let rstd_1 = T::load(rstd_ptr.add_offsets(row_idx),   None, None, &[], None, None, None, false);
    let rstd   = T::broadcast_to(rstd_1, &[BLOCK_N]);

    // Pass 1: Σ dy · γ · h
    let mut dot = zero1;
    let mut n_start: i32 = 0;
    while n_start < N {
        let col  = T::arange(0, BLOCK_N) + n_start;
        let mask = col.lt(N);

        let dy = T::load(dy_ptr.add_offsets(col + row_start), Some(mask), Some(zeros), &[], None, None, None, false);
        let hi = T::load(h_ptr.add_offsets(col + row_start),  Some(mask), Some(zeros), &[], None, None, None, false);
        let wi = T::load(weight_ptr.add_offsets(col),          Some(mask), Some(zeros), &[], None, None, None, false);
        dot = dot + T::sum(dy * wi * hi, None, true);
        n_start += BLOCK_N;
    }

    let rstd_sq = T::broadcast_to(rstd_1 * rstd_1, &[BLOCK_N]);
    let scale   = T::broadcast_to(dot * n_inv, &[BLOCK_N]);

    // Pass 2: dx, dresidual, dweight
    n_start = 0;
    while n_start < N {
        let col  = T::arange(0, BLOCK_N) + n_start;
        let mask = col.lt(N);

        let dy   = T::load(dy_ptr.add_offsets(col + row_start),   Some(mask), Some(zeros), &[], None, None, None, false);
        let hi   = T::load(h_ptr.add_offsets(col + row_start),    Some(mask), Some(zeros), &[], None, None, None, false);
        let wi   = T::load(weight_ptr.add_offsets(col),            Some(mask), Some(zeros), &[], None, None, None, false);
        let dw_old = T::load(dweight_ptr.add_offsets(col),         Some(mask), Some(zeros), &[], None, None, None, false);

        let grad = rstd * wi * (dy - hi * rstd_sq * scale);
        T::store(dx_ptr.add_offsets(col + row_start),        grad, Some(mask), &[], None, None);
        T::store(dresidual_ptr.add_offsets(col + row_start), grad, Some(mask), &[], None, None);
        T::store(dweight_ptr.add_offsets(col), dw_old + dy * hi * rstd, Some(mask), &[], None, None);
        n_start += BLOCK_N;
    }
}

// ── RuntimeOp — forward ───────────────────────────────────────────────────────

/// RuntimeOp wrapper — carries `eps` alongside the generated kernel struct.
pub struct FusedAddRmsnormRtOp {
    pub forward: FusedAddRmsnormForward<f32>,
    pub eps:     f32,
}

/// `param_shapes` returns `[weight[N], h_cache[M,N], rstd_cache[M]]`.
///
/// Slots 1 and 2 (`h_cache`, `rstd_cache`) are activation caches written by
/// the forward kernel and read by the backward kernel.  They are allocated by
/// the executor but are NOT loaded from a weight file.
impl RuntimeOp for FusedAddRmsnormRtOp {
    fn n_activation_inputs(&self) -> usize { 2 } // x, residual

    fn param_shapes(
        &self,
        input_shapes: &[&[usize]],
        output_shape: &[usize],
    ) -> Vec<Vec<usize>> {
        let m = output_shape[0];
        let n = output_shape[1];
        let _ = input_shapes;
        vec![
            vec![n],    // params[0] = weight (γ)
            vec![m, n], // params[1] = h_cache  (x + residual, for bwd)
            vec![m],    // params[2] = rstd_cache
        ]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight"]  // only slot 0 is a named/loadable weight
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
        let m = output_shape[0] as i32;
        let n = output_shape[1] as i32;
        // Kernel: (x, residual, y, h, rstd, weight, M, N, eps)
        visitor.visit_ptr(inputs[0].0); // x_ptr
        visitor.visit_ptr(inputs[1].0); // residual_ptr
        visitor.visit_ptr(output);       // y_ptr
        visitor.visit_ptr(params[1]);    // h_ptr   (cache)
        visitor.visit_ptr(params[2]);    // rstd_ptr (cache)
        visitor.visit_ptr(params[0]);    // weight_ptr (γ)
        visitor.visit_i32(m);
        visitor.visit_i32(n);
        visitor.visit_f32(self.eps);
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
        _inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        _output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let m = output_shape[0] as i32;
        let n = output_shape[1] as i32;
        // Kernel: (dy, h, dx, dresidual, dweight, weight, rstd, M, N)
        visitor.visit_ptr(grad_output);      // dy_ptr
        visitor.visit_ptr(params[1]);        // h_ptr  (cached h = x + residual)
        visitor.visit_ptr(grad_inputs[0]);   // dx_ptr
        visitor.visit_ptr(grad_inputs[1]);   // dresidual_ptr
        visitor.visit_ptr(grad_params[0]);   // dweight_ptr
        visitor.visit_ptr(params[0]);        // weight_ptr
        visitor.visit_ptr(params[2]);        // rstd_ptr
        visitor.visit_i32(m);
        visitor.visit_i32(n);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        [output_shape[0] as u32, 1, 1]
    }
}

// ── CustomOp — graph wrapper ──────────────────────────────────────────────────

/// Graph node for fused residual-add + RMSNorm.
///
/// Inputs:  `x [M, N]` (primary), `residual [M, N]` (other)
/// Output:  `y [M, N]`  (normalised)
///
/// The updated residual `h = x + residual` and `rstd` are stored in param
/// slots 1 and 2 and are accessible to the backward kernel.
pub struct FusedAddRmsnormOp {
    pub block_n: i32,
    pub eps: f32,
}

impl FusedAddRmsnormOp {
    pub fn new(eps: f32) -> Self {
        Self { block_n: 1024, eps }
    }

    pub fn custom_data(eps: f32) -> CustomData {
        CustomData::new(Self::new(eps))
    }
}

impl CustomOp for FusedAddRmsnormOp {
    fn name(&self) -> &str { "fused_add_rmsnorm" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let forward = FusedAddRmsnormForward::<f32>::new(self.block_n);
        let name   = forward.name.to_string();
        let source = forward.source.clone();
        let rop = FusedAddRmsnormRtOp { forward, eps: self.eps };
        Some((name, source, "entry_point".to_string(), Arc::new(rop)))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        FusedAddRmsnormBackward::<f32>::new(self.block_n).source.clone()
    }
}
