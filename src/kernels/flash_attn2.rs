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

//! Flash Attention 2 custom op wrapper for the transformer graph.
//!
//! Wraps the `flash_attention2_forward` kernel from `teeny-kernels` and adds
//! a combined single-pass backward that recomputes attention weights and
//! accumulates `dK`/`dV` via atomic-add (one CTA per query row).
//!
//! **Layout**: `[BH, N_CTX, HEAD_DIM]` where `BH = BATCH × N_HEADS`.
//!
//! **Forward**
//! - Inputs (activation): `q [BH, Nq, D]`, `k [BH, Nk, D]`, `v [BH, Nk, D]`
//! - Output: `o [BH, Nq, D]`
//! - Param cache: `l [BH, Nq]` (log-sum-exp, saved for backward)
//!
//! **Backward** — grid `[Nq, BH, 1]`; one CTA per query row.
//! Each CTA computes `dQ[q]` (exact) and atomically accumulates into
//! `dK[k]`, `dV[k]` for every `k` in the inner loop.  Works for both
//! self-attention (`Nq = Nk`) and cross-attention (`Nq ≠ Nk`).
//!
//! `dK` and `dV` must be zero-initialised by the executor before this kernel.

use core::any::Any;
use std::sync::Arc;

use teeny_core::{
    device::program::ArgVisitor,
    graph::{CustomData, CustomOp, Shape},
    model::RuntimeOp,
};
use teeny_kernels::nn::attention::flash_attn2::FlashAttention2Forward;
use teeny_macros::kernel;
use teeny_triton::triton::{
    Axis,
    types::{AddOffsets, Comparison, Tensor},
    *,
};

// ── Backward ─────────────────────────────────────────────────────────────────

/// Causal Flash Attention 2 forward (decoder self-attention).
///
/// Identical to `flash_attention2_forward` from teeny-kernels except that
/// future key positions (`k_row > pid_m`) are masked to `neg_inf`.
///
/// Grid: `(N_CTX_Q, BH, 1)`.
#[kernel]
pub fn flash_attn2_causal_fwd<T: Triton, const HEAD_DIM: i32>(
    q_ptr: T::Pointer<f32>,
    k_ptr: T::Pointer<f32>,
    v_ptr: T::Pointer<f32>,
    o_ptr: T::Pointer<f32>,
    l_ptr: T::Pointer<f32>,
    n_ctx_q: i32,
    n_ctx_k: i32,
    softmax_scale: f32,
    neg_inf: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_m  = T::program_id(Axis::X);
    let pid_bh = T::program_id(Axis::Y);

    let kv_bh_base = pid_bh * n_ctx_k * HEAD_DIM;
    let q_row_base = pid_bh * n_ctx_q * HEAD_DIM + pid_m * HEAD_DIM;
    let l_row_base = pid_bh * n_ctx_q + pid_m;

    let d       = T::arange(0, HEAD_DIM);
    let scale_t = T::full::<f32>(&[HEAD_DIM], softmax_scale);
    let neg_inf_t = T::full::<f32>(&[HEAD_DIM], neg_inf);

    let q_vec = T::load(q_ptr.add_offsets(d + q_row_base), None, None, &[], None, None, None, false);

    let mut acc = T::zeros::<f32>(&[HEAD_DIM]);
    let mut m_i = T::full::<f32>(&[HEAD_DIM], neg_inf);
    let mut l_i = T::zeros::<f32>(&[HEAD_DIM]);

    let mut k_row: i32 = 0;
    while k_row < n_ctx_k {
        let kv_row_base = kv_bh_base + k_row * HEAD_DIM;

        let k_vec = T::load(k_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);
        let v_vec = T::load(v_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);

        let qk = T::sum(q_vec * k_vec, Some(0), true) * scale_t;

        // Causal mask: future positions get -inf
        let future_pos = T::broadcast_to(T::full::<i32>(&[1], k_row), &[HEAD_DIM]);
        let cur_pos    = T::broadcast_to(T::full::<i32>(&[1], pid_m), &[HEAD_DIM]);
        let is_future  = T::gt::<i32>(future_pos, cur_pos);
        let qk_masked  = T::where_(is_future, neg_inf_t, qk);

        let m_new    = T::maximum(m_i, qk_masked);
        let exp_diff = T::exp(m_i - m_new);
        let p        = T::exp(qk_masked - m_new);

        l_i = exp_diff * l_i + p;
        acc = exp_diff * acc + p * v_vec;
        m_i = m_new;

        k_row += 1;
    }

    let o_row       = acc / l_i;
    let l_save_sum  = T::sum(m_i + T::log(l_i), Some(0), false);
    let l_save      = l_save_sum / T::full::<f32>(&[1], HEAD_DIM as f32);

    T::store(o_ptr.add_offsets(d + q_row_base), o_row, None, &[], None, None);
    T::store(l_ptr.add_offsets(T::arange(0, 1) + l_row_base), l_save, None, &[], None, None);
}

// ── Causal backward ───────────────────────────────────────────────────────────

/// Combined causal backward — same as non-causal but skips future keys in dQ.
/// For dK/dV accumulation (via atomics), future positions contribute zero gradient
/// since their attention weight was -inf → p = 0.
///
/// Grid: `(Nq, BH, 1)`.
#[cfg(feature = "training")]
#[kernel]
pub fn flash_attn2_causal_bwd<T: Triton, const HEAD_DIM: i32>(
    q_ptr:  T::Pointer<f32>,
    k_ptr:  T::Pointer<f32>,
    v_ptr:  T::Pointer<f32>,
    o_ptr:  T::Pointer<f32>,
    do_ptr: T::Pointer<f32>,
    l_ptr:  T::Pointer<f32>,
    dq_ptr: T::Pointer<f32>,
    dk_ptr: T::Pointer<f32>,
    dv_ptr: T::Pointer<f32>,
    n_ctx_q: i32,
    n_ctx_k: i32,
    softmax_scale: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_q  = T::program_id(Axis::X);
    let pid_bh = T::program_id(Axis::Y);

    let q_row_base = pid_bh * n_ctx_q * HEAD_DIM + pid_q * HEAD_DIM;
    let kv_bh_base = pid_bh * n_ctx_k * HEAD_DIM;
    let l_row_base = pid_bh * n_ctx_q + pid_q;

    let d       = T::arange(0, HEAD_DIM);
    let scale   = T::full::<f32>(&[HEAD_DIM], softmax_scale);
    let zeros_t = T::zeros::<f32>(&[HEAD_DIM]);

    let q_vec  = T::load(q_ptr.add_offsets(d + q_row_base),  None, None, &[], None, None, None, false);
    let o_vec  = T::load(o_ptr.add_offsets(d + q_row_base),  None, None, &[], None, None, None, false);
    let do_vec = T::load(do_ptr.add_offsets(d + q_row_base), None, None, &[], None, None, None, false);
    let l_raw  = T::load(l_ptr.add_offsets(T::arange(0, 1) + l_row_base), None, None, &[], None, None, None, false);
    let l_q    = T::broadcast_to(T::sum(l_raw, Some(0), false), &[HEAD_DIM]);
    let d_q    = T::broadcast_to(T::sum(o_vec * do_vec, Some(0), false), &[HEAD_DIM]);

    let mut dq_acc = T::zeros::<f32>(&[HEAD_DIM]);

    let mut k_row: i32 = 0;
    while k_row < n_ctx_k {
        let kv_row_base = kv_bh_base + k_row * HEAD_DIM;

        let future_pos = T::broadcast_to(T::full::<i32>(&[1], k_row), &[HEAD_DIM]);
        let cur_pos    = T::broadcast_to(T::full::<i32>(&[1], pid_q), &[HEAD_DIM]);
        let is_future  = T::gt::<i32>(future_pos, cur_pos);

        let k_vec = T::load(k_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);
        let v_vec = T::load(v_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);

        let qk = T::broadcast_to(T::sum(q_vec * k_vec, Some(0), false), &[HEAD_DIM]) * scale;
        // Zero out future positions
        let qk_m = T::where_(is_future, T::broadcast_to(T::full::<f32>(&[1], f32::NEG_INFINITY), &[HEAD_DIM]), qk);
        let p    = T::where_(is_future, zeros_t, T::exp(qk_m - l_q));

        let do_dot_v = T::broadcast_to(T::sum(do_vec * v_vec, Some(0), false), &[HEAD_DIM]);
        let ds = p * (do_dot_v - d_q);

        dq_acc = dq_acc + ds * k_vec * scale;

        T::atomic_add(dv_ptr.add_offsets(d + kv_row_base), p * do_vec, None, None, None);
        T::atomic_add(dk_ptr.add_offsets(d + kv_row_base), ds * q_vec * scale, None, None, None);

        k_row += 1;
    }

    T::store(dq_ptr.add_offsets(d + q_row_base), dq_acc, None, &[], None, None);
}

/// Flash Attention 2 combined backward.
///
/// Computes `dQ`, and atomically accumulates `dK` + `dV`.
/// Grid: `(Nq, BH, 1)`.
#[cfg(feature = "training")]
#[kernel]
pub fn flash_attn2_backward<T: Triton, const HEAD_DIM: i32>(
    q_ptr:  T::Pointer<f32>,
    k_ptr:  T::Pointer<f32>,
    v_ptr:  T::Pointer<f32>,
    o_ptr:  T::Pointer<f32>,
    do_ptr: T::Pointer<f32>,
    l_ptr:  T::Pointer<f32>,
    dq_ptr: T::Pointer<f32>,
    dk_ptr: T::Pointer<f32>,
    dv_ptr: T::Pointer<f32>,
    n_ctx_q: i32,
    n_ctx_k: i32,
    softmax_scale: f32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid_q  = T::program_id(Axis::X);  // query-row index [0, n_ctx_q)
    let pid_bh = T::program_id(Axis::Y);  // (batch, head)   [0, BH)

    let q_row_base = pid_bh * n_ctx_q * HEAD_DIM + pid_q * HEAD_DIM;
    let kv_bh_base = pid_bh * n_ctx_k * HEAD_DIM;
    let l_row_base = pid_bh * n_ctx_q + pid_q;

    let d      = T::arange(0, HEAD_DIM);
    let scale  = T::full::<f32>(&[HEAD_DIM], softmax_scale);

    // Load Q[q], O[q], dO[q], L[q] for this CTA's query row.
    let q_vec  = T::load(q_ptr.add_offsets(d + q_row_base),  None, None, &[], None, None, None, false);
    let o_vec  = T::load(o_ptr.add_offsets(d + q_row_base),  None, None, &[], None, None, None, false);
    let do_vec = T::load(do_ptr.add_offsets(d + q_row_base), None, None, &[], None, None, None, false);
    let l_raw  = T::load(l_ptr.add_offsets(T::arange(0, 1) + l_row_base), None, None, &[], None, None, None, false);
    let l_q    = T::broadcast_to(T::sum(l_raw, Some(0), false), &[HEAD_DIM]);

    // D_q = rowsum(O_q · dO_q) — scalar replicated to [HEAD_DIM].
    let d_q = T::broadcast_to(T::sum(o_vec * do_vec, Some(0), false), &[HEAD_DIM]);

    let mut dq_acc = T::zeros::<f32>(&[HEAD_DIM]);

    let mut k_row: i32 = 0;
    while k_row < n_ctx_k {
        let kv_row_base = kv_bh_base + k_row * HEAD_DIM;

        let k_vec = T::load(k_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);
        let v_vec = T::load(v_ptr.add_offsets(d + kv_row_base), None, None, &[], None, None, None, false);

        // Recompute p_{qk} = exp(Q_q · K_k * scale - L_q) — [HEAD_DIM] (all elements equal scalar)
        let qk   = T::broadcast_to(T::sum(q_vec * k_vec, Some(0), false), &[HEAD_DIM]) * scale;
        let p    = T::exp(qk - l_q);

        // dS_{qk} = p * (dO_q · V_k - D_q)
        let do_dot_v = T::broadcast_to(T::sum(do_vec * v_vec, Some(0), false), &[HEAD_DIM]);
        let ds = p * (do_dot_v - d_q);

        // dQ += ds * K_k * scale
        dq_acc = dq_acc + ds * k_vec * scale;

        // dV[k] += p * dO_q  (atomic — multiple Q-row CTAs write here)
        T::atomic_add(dv_ptr.add_offsets(d + kv_row_base), p * do_vec, None, None, None);

        // dK[k] += ds * Q_q * scale  (atomic)
        T::atomic_add(dk_ptr.add_offsets(d + kv_row_base), ds * q_vec * scale, None, None, None);

        k_row += 1;
    }

    T::store(dq_ptr.add_offsets(d + q_row_base), dq_acc, None, &[], None, None);
}

// ── RuntimeOp ────────────────────────────────────────────────────────────────

/// RuntimeOp wrapping `FlashAttention2Forward` from teeny-kernels.
pub struct FlashAttn2RuntimeOp {
    softmax_scale: f32,
}

impl FlashAttn2RuntimeOp {
    pub fn new(_head_dim: i32, softmax_scale: f32) -> Self {
        Self { softmax_scale }
    }
}

impl RuntimeOp for FlashAttn2RuntimeOp {
    /// 3 activation inputs: Q, K, V
    fn n_activation_inputs(&self) -> usize { 3 }

    fn param_shapes(
        &self,
        input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> Vec<Vec<usize>> {
        // l_cache [BH, Nq]
        let bh = input_shapes[0][0];
        let nq = input_shapes[0][1];
        vec![vec![bh, nq]]
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
        let bh   = output_shape[0] as i32;
        let nq   = output_shape[1] as i32;
        let nk   = inputs[1].1[1] as i32;
        let _ = bh;
        // Kernel: (q, k, v, o, l, n_ctx_q, n_ctx_k, softmax_scale, neg_inf)
        visitor.visit_ptr(inputs[0].0);    // q_ptr
        visitor.visit_ptr(inputs[1].0);    // k_ptr
        visitor.visit_ptr(inputs[2].0);    // v_ptr
        visitor.visit_ptr(output);          // o_ptr
        visitor.visit_ptr(params[0]);       // l_ptr (cache)
        visitor.visit_i32(nq);
        visitor.visit_i32(nk);
        visitor.visit_f32(self.softmax_scale);
        visitor.visit_f32(f32::NEG_INFINITY);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let bh = output_shape[0] as u32;
        let nq = output_shape[1] as u32;
        [nq, bh, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let nq = output_shape[1] as i32;
        let nk = inputs[1].1[1] as i32;
        // Kernel: (q, k, v, o, do, l, dq, dk, dv, n_ctx_q, n_ctx_k, softmax_scale)
        visitor.visit_ptr(inputs[0].0);    // q_ptr
        visitor.visit_ptr(inputs[1].0);    // k_ptr
        visitor.visit_ptr(inputs[2].0);    // v_ptr
        visitor.visit_ptr(output);          // o_ptr
        visitor.visit_ptr(grad_output);     // do_ptr
        visitor.visit_ptr(params[0]);       // l_ptr
        visitor.visit_ptr(grad_inputs[0]);  // dq_ptr
        visitor.visit_ptr(grad_inputs[1]);  // dk_ptr
        visitor.visit_ptr(grad_inputs[2]);  // dv_ptr
        visitor.visit_i32(nq);
        visitor.visit_i32(nk);
        visitor.visit_f32(self.softmax_scale);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let bh = output_shape[0] as u32;
        let nq = output_shape[1] as u32;
        [nq, bh, 1]
    }
}

// ── CustomOp ────────────────────────────────────────────────────────────────

/// Graph node for Flash Attention 2.
///
/// Primary input: `q [BH, Nq, HEAD_DIM]`
/// Other inputs:  `k [BH, Nk, HEAD_DIM]`, `v [BH, Nk, HEAD_DIM]`
/// Output:        `o [BH, Nq, HEAD_DIM]`
pub struct FlashAttn2Op {
    pub head_dim:      i32,
    pub softmax_scale: f32,
}

impl FlashAttn2Op {
    pub fn new(head_dim: usize, softmax_scale: f32) -> Self {
        Self { head_dim: head_dim as i32, softmax_scale }
    }

    pub fn custom_data(head_dim: usize, softmax_scale: f32) -> CustomData {
        CustomData::new(Self::new(head_dim, softmax_scale))
    }
}

impl CustomOp for FlashAttn2Op {
    fn name(&self) -> &str { "flash_attn2" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let inner = FlashAttention2Forward::new(self.head_dim);
        let name   = inner.name.to_string();
        let source = inner.source.clone();
        let rop    = FlashAttn2RuntimeOp::new(self.head_dim, self.softmax_scale);
        Some((name, source, "entry_point".to_string(), Arc::new(rop)))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        FlashAttn2Backward::new(self.head_dim).source.clone()
    }
}

// ── Causal RuntimeOp ─────────────────────────────────────────────────────────

/// RuntimeOp for causal Flash Attention 2 (decoder self-attention).
///
/// Identical arg layout to [`FlashAttn2RuntimeOp`] but uses the causal kernels.
pub struct CausalFlashAttn2RuntimeOp {
    softmax_scale: f32,
}

impl CausalFlashAttn2RuntimeOp {
    pub fn new(_head_dim: i32, softmax_scale: f32) -> Self {
        Self { softmax_scale }
    }
}

impl RuntimeOp for CausalFlashAttn2RuntimeOp {
    fn n_activation_inputs(&self) -> usize { 3 } // Q, K, V

    fn param_shapes(
        &self,
        input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> Vec<Vec<usize>> {
        let bh = input_shapes[0][0];
        let nq = input_shapes[0][1];
        vec![vec![bh, nq]] // l_cache
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
        let nq = output_shape[1] as i32;
        let nk = inputs[1].1[1] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(inputs[2].0);
        visitor.visit_ptr(output);
        visitor.visit_ptr(params[0]);
        visitor.visit_i32(nq);
        visitor.visit_i32(nk);
        visitor.visit_f32(self.softmax_scale);
        visitor.visit_f32(f32::NEG_INFINITY);
    }

    fn block(&self) -> [u32; 3] { [128, 1, 1] }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        let bh = output_shape[0] as u32;
        let nq = output_shape[1] as u32;
        [nq, bh, 1]
    }

    #[cfg(feature = "training")]
    fn has_backward(&self) -> bool { true }

    #[cfg(feature = "training")]
    fn pack_backward_args(
        &self,
        inputs: &[(teeny_core::model::RawPtr, &[usize])],
        params: &[teeny_core::model::RawPtr],
        output: teeny_core::model::RawPtr,
        output_shape: &[usize],
        grad_output: teeny_core::model::RawPtr,
        _grad_output_row_stride: i32,
        grad_inputs: &[teeny_core::model::RawPtr],
        _grad_params: &[teeny_core::model::RawPtr],
        visitor: &mut dyn ArgVisitor,
    ) {
        let nq = output_shape[1] as i32;
        let nk = inputs[1].1[1] as i32;
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(inputs[2].0);
        visitor.visit_ptr(output);
        visitor.visit_ptr(grad_output);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(grad_inputs[0]);
        visitor.visit_ptr(grad_inputs[1]);
        visitor.visit_ptr(grad_inputs[2]);
        visitor.visit_i32(nq);
        visitor.visit_i32(nk);
        visitor.visit_f32(self.softmax_scale);
    }

    #[cfg(feature = "training")]
    fn backward_block(&self) -> [u32; 3] { [128, 1, 1] }

    #[cfg(feature = "training")]
    fn backward_grid(&self, _input_shapes: &[&[usize]], output_shape: &[usize]) -> [u32; 3] {
        let bh = output_shape[0] as u32;
        let nq = output_shape[1] as u32;
        [nq, bh, 1]
    }
}

// ── CausalFlashAttn2Op ───────────────────────────────────────────────────────

/// Graph node for causal Flash Attention 2 (decoder self-attention).
///
/// Primary input: `q [BH, Nq, HEAD_DIM]`
/// Other inputs:  `k [BH, Nk, HEAD_DIM]`, `v [BH, Nk, HEAD_DIM]`
/// Output:        `o [BH, Nq, HEAD_DIM]`
pub struct CausalFlashAttn2Op {
    pub head_dim:      i32,
    pub softmax_scale: f32,
}

impl CausalFlashAttn2Op {
    pub fn new(head_dim: usize, softmax_scale: f32) -> Self {
        Self { head_dim: head_dim as i32, softmax_scale }
    }

    pub fn custom_data(head_dim: usize, softmax_scale: f32) -> CustomData {
        CustomData::new(Self::new(head_dim, softmax_scale))
    }
}

impl CustomOp for CausalFlashAttn2Op {
    fn name(&self) -> &str { "causal_flash_attn2" }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].clone()
    }

    fn as_any(&self) -> &dyn Any { self }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = FlashAttn2CausalFwd::new(self.head_dim);
        let name   = kernel.name.to_string();
        let src    = kernel.source.clone();
        let rop    = CausalFlashAttn2RuntimeOp::new(self.head_dim, self.softmax_scale);
        Some((name, src, "entry_point".to_string(), Arc::new(rop)))
    }

    #[cfg(feature = "training")]
    fn lower_backward_source(&self) -> String {
        FlashAttn2CausalBwd::new(self.head_dim).source.clone()
    }
}
