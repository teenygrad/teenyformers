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

//! Token embedding gather kernel.
//!
//! One CTA per sequence position.  Each CTA loads `ids[pid]` (stored as f32),
//! converts to i32, then copies the corresponding row of the embedding table
//! to the output buffer in tiles of `BLOCK_D` elements.

use teeny_macros::kernel;
use teeny_triton::triton::{
    Axis,
    types::{AddOffsets, Comparison, Tensor},
    *,
};

/// Token embedding forward: `out[s, :] = weight[ids[s], :]`.
///
/// - `ids_ptr`    — `[S]` sequence of token IDs, **stored as f32**.
/// - `weight_ptr` — `[vocab_size, d_model]` embedding table.
/// - `out_ptr`    — `[S, d_model]` output embeddings.
///
/// Grid: `[S, 1, 1]` — one CTA per token position.
/// `BLOCK_D` must be a power of two ≥ 1.
#[kernel]
pub fn token_embed_forward<T: Triton, const BLOCK_D: i32>(
    ids_ptr:    T::Pointer<f32>,
    weight_ptr: T::Pointer<f32>,
    out_ptr:    T::Pointer<f32>,
    _seq_len:   i32,
    d_model:    i32,
) where
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);

    // Load the token ID for this position (f32-encoded integer).
    let token_id = T::load_scalar_f32_as_i32(ids_ptr, pid);

    let embed_base = token_id * d_model; // row start in weight table
    let out_base   = pid      * d_model; // row start in output buffer

    let zeros = T::zeros::<f32>(&[BLOCK_D]);

    let mut d_start: i32 = 0;
    while d_start < d_model {
        let col  = T::arange(0, BLOCK_D) + d_start;
        let mask = col.lt(d_model);

        let row = T::load(
            weight_ptr.add_offsets(col + embed_base),
            Some(mask), Some(zeros), &[], None, None, None, false,
        );
        T::store(
            out_ptr.add_offsets(col + out_base),
            row, Some(mask), &[], None, None,
        );

        d_start += BLOCK_D;
    }
}
