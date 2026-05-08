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

//! Rotary Position Embedding (RoPE) kernels.
//!
//! Layout: `[BH, N_CTX, HEAD_DIM]` row-major, where `BH = BATCH * N_HEADS`.
//!
//! **Algorithm**: for token at position `pos`, pair `k ∈ [0, HEAD_DIM/2)`:
//! ```text
//! θ_k      = pos · base^(-2k/HEAD_DIM)
//! y[2k]    =  x[2k]   · cos(θ_k) − x[2k+1] · sin(θ_k)
//! y[2k+1]  =  x[2k]   · sin(θ_k) + x[2k+1] · cos(θ_k)
//! ```
//!
//! **Backward**: rotation is orthogonal (R⁻¹ = Rᵀ), so the backward is the
//! inverse rotation:
//! ```text
//! dx[2k]   =  dy[2k] · cos(θ_k) + dy[2k+1] · sin(θ_k)
//! dx[2k+1] = -dy[2k] · sin(θ_k) + dy[2k+1] · cos(θ_k)
//! ```
//!
//! Grid: `(N_CTX, BH, 1)` — one CTA per `(batch_head, position)` pair.

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

/// RoPE forward pass.
///
/// Grid: `(N_CTX, BH, 1)`.  `HEAD_DIM` must be even and a power of two.
#[kernel]
pub fn rope_forward<T: Triton, const HEAD_DIM: i32>(
    x_ptr:        T::Pointer<f32>,
    y_ptr:        T::Pointer<f32>,
    n_ctx:        i32,
    ln_rope_base: f32,  // ln(rope_base), pre-computed by caller
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_pos = T::program_id(Axis::X); // token position [0, N_CTX)
    let pid_bh = T::program_id(Axis::Y);  // batch * head   [0, BH)

    let row_base = pid_bh * n_ctx * HEAD_DIM + pid_pos * HEAD_DIM;

    // Stride-2 offsets for the two interleaved halves.
    // HEAD_DIM is a const generic so HEAD_DIM/2 folds to a compile-time
    // constant, avoiding a named MIR local that teenyc cannot copy.
    let k = T::arange(0, HEAD_DIM / 2);     // [0, 1, …, half-1]  I32Tensor
    let even_offs = k * 2 + row_base;       // x[2k]
    let odd_offs = k * 2 + 1 + row_base;   // x[2k+1]

    let x_even = T::load(x_ptr.add_offsets(even_offs), None, None, &[], None, None, None, false);
    let x_odd  = T::load(x_ptr.add_offsets(odd_offs),  None, None, &[], None, None, None, false);

    // θ_k = pos · exp(-k · 2·ln(base) / HEAD_DIM)
    // All sub-expressions are inlined (no named f32 locals reused) so the
    // teenyc MLIR backend never emits a MIR `copy` node.
    let cos_t = T::cos(
        T::full::<f32>(&[HEAD_DIM / 2], pid_pos as f32)
            * T::exp(T::arange_f32(0, HEAD_DIM / 2) * T::full::<f32>(&[HEAD_DIM / 2], -2.0_f32 * ln_rope_base / HEAD_DIM as f32))
    );
    let sin_t = T::sin(
        T::full::<f32>(&[HEAD_DIM / 2], pid_pos as f32)
            * T::exp(T::arange_f32(0, HEAD_DIM / 2) * T::full::<f32>(&[HEAD_DIM / 2], -2.0_f32 * ln_rope_base / HEAD_DIM as f32))
    );

    T::store(y_ptr.add_offsets(even_offs), x_even * cos_t - x_odd * sin_t, None, &[], None, None);
    T::store(y_ptr.add_offsets(odd_offs),  x_even * sin_t + x_odd * cos_t, None, &[], None, None);
}

// ── Backward ─────────────────────────────────────────────────────────────────

/// RoPE backward pass — inverse (transposed) rotation.
///
/// Grid: `(N_CTX, BH, 1)` — same as forward.
#[cfg(feature = "training")]
#[kernel]
pub fn rope_backward<T: Triton, const HEAD_DIM: i32>(
    dy_ptr:       T::Pointer<f32>,
    dx_ptr:       T::Pointer<f32>,
    n_ctx:        i32,
    ln_rope_base: f32,  // ln(rope_base), pre-computed by caller
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_pos = T::program_id(Axis::X);
    let pid_bh  = T::program_id(Axis::Y);

    let row_base = pid_bh * n_ctx * HEAD_DIM + pid_pos * HEAD_DIM;

    let k = T::arange(0, HEAD_DIM / 2);
    let even_offs = k * 2 + row_base;
    let odd_offs  = k * 2 + 1 + row_base;

    let dy_even = T::load(dy_ptr.add_offsets(even_offs), None, None, &[], None, None, None, false);
    let dy_odd  = T::load(dy_ptr.add_offsets(odd_offs),  None, None, &[], None, None, None, false);

    let cos_t = T::cos(
        T::full::<f32>(&[HEAD_DIM / 2], pid_pos as f32)
            * T::exp(T::arange_f32(0, HEAD_DIM / 2) * T::full::<f32>(&[HEAD_DIM / 2], -2.0_f32 * ln_rope_base / HEAD_DIM as f32))
    );
    let sin_t = T::sin(
        T::full::<f32>(&[HEAD_DIM / 2], pid_pos as f32)
            * T::exp(T::arange_f32(0, HEAD_DIM / 2) * T::full::<f32>(&[HEAD_DIM / 2], -2.0_f32 * ln_rope_base / HEAD_DIM as f32))
    );

    // Inverse rotation: R(-θ) applied to dy.
    T::store(dx_ptr.add_offsets(even_offs),  dy_even * cos_t + dy_odd * sin_t, None, &[], None, None);
    T::store(dx_ptr.add_offsets(odd_offs),  -dy_even * sin_t + dy_odd * cos_t, None, &[], None, None);
}

// ── RuntimeOp — forward ───────────────────────────────────────────────────────

/// RuntimeOp wrapper that stores `rope_base` alongside the generated kernel struct.
///
/// `RopeForward` (generated by `#[kernel]`) only carries `HEAD_DIM`; runtime
/// parameters such as `rope_base` must live in this outer wrapper.
pub struct RopeRtOp {
    pub forward:   RopeForward,
    pub rope_base: f32,
    #[cfg(feature = "training")]
    pub backward_src: String,
}

impl RopeRtOp {
    pub fn new(head_dim: i32, rope_base: f32) -> Self {
        let forward = RopeForward::new(head_dim);
        #[cfg(feature = "training")]
        let backward_src = RopeBackward::new(head_dim).source.clone();
        Self {
            forward,
            rope_base,
            #[cfg(feature = "training")]
            backward_src,
        }
    }
}

impl RuntimeOp for RopeRtOp {
    fn n_activation_inputs(&self) -> usize { 1 }

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
        // x_ptr, y_ptr, n_ctx, ln_rope_base
        let n_ctx = output_shape[1] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(n_ctx);
        visitor.visit_f32(self.rope_base.ln());
    }

    // Grid: (N_CTX, BH, 1)
    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let bh    = output_shape[0] as u32;
        let n_ctx = output_shape[1] as u32;
        [n_ctx, bh, 1]
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
        let n_ctx = output_shape[1] as i32;
        visitor.visit_ptr(grad_output);    // dy_ptr
        visitor.visit_ptr(grad_inputs[0]); // dx_ptr
        visitor.visit_i32(n_ctx);
        visitor.visit_f32(self.rope_base.ln());
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let bh    = output_shape[0] as u32;
        let n_ctx = output_shape[1] as u32;
        [n_ctx, bh, 1]
    }
}

// ── CustomOp — graph wrapper ──────────────────────────────────────────────────

/// Graph node wrapping the RoPE forward kernel.
///
/// Input:  `x [BH, N_CTX, HEAD_DIM]`
/// Output: `y [BH, N_CTX, HEAD_DIM]`  (same shape)
pub struct RopeOp {
    pub head_dim: i32,
    pub rope_base: f32,
}

impl RopeOp {
    pub fn new(head_dim: usize, rope_base: f32) -> Self {
        Self { head_dim: head_dim as i32, rope_base }
    }

    pub fn custom_data(head_dim: usize, rope_base: f32) -> CustomData {
        CustomData::new(Self::new(head_dim, rope_base))
    }
}

impl CustomOp for RopeOp {
    fn name(&self) -> &str { "rope" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let rop = RopeRtOp::new(self.head_dim, self.rope_base);
        let name   = rop.forward.name.to_string();
        let source = rop.forward.source.clone();
        Some((name, source, "entry_point".to_string(), Arc::new(rop)))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        RopeBackward::new(self.head_dim).source.clone()
    }
}
