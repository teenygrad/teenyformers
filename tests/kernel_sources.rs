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

//! Source snapshot tests for teenyformers custom kernels.
//!
//! Each test instantiates a kernel struct and records its Triton IR source
//! as an insta snapshot.  No GPU or CUDA installation is required — these
//! are pure host-side checks that the code-generation is stable.

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;
use teeny_cuda::compiler::target::Capability;

const BLOCK_N:   i32 = 16;
const HEAD_DIM:  i32 = 64;

fn sm90_target() -> Target {
    Target::new(Capability::Sm90)
}

const BLOCK_D: i32 = 16;

// ── TokenEmbed ────────────────────────────────────────────────────────────────

#[test]
fn test_token_embed_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::token_embed::TokenEmbedForward::new(BLOCK_D);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("token_embed_forward_source", kernel.source());
    Ok(())
}

// ── SwiGLU ───────────────────────────────────────────────────────────────────

#[test]
fn test_swiglu_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::swiglu::SwigluForward::<f32>::new(BLOCK_N);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("swiglu_forward_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_swiglu_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::swiglu::SwigluBackward::<f32>::new(BLOCK_N);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("swiglu_backward_source", kernel.source());
    Ok(())
}

// ── FusedAddRmsnorm ───────────────────────────────────────────────────────────

#[test]
fn test_fused_add_rmsnorm_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::fused_add_rmsnorm::FusedAddRmsnormForward::<f32>::new(BLOCK_N);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("fused_add_rmsnorm_forward_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_fused_add_rmsnorm_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::fused_add_rmsnorm::FusedAddRmsnormBackward::<f32>::new(BLOCK_N);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("fused_add_rmsnorm_backward_source", kernel.source());
    Ok(())
}

// ── RoPE ─────────────────────────────────────────────────────────────────────

#[test]
fn test_rope_forward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::rope::RopeForward::new(HEAD_DIM);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("rope_forward_source", kernel.source());
    Ok(())
}

#[cfg(feature = "training")]
#[test]
fn test_rope_backward_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::rope::RopeBackward::new(HEAD_DIM);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("rope_backward_source", kernel.source());
    Ok(())
}

// ── TransposeHeads / MergeHeads ──────────────────────────────────────────────

#[test]
fn test_transpose_heads_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::transpose_heads::TransposeHeads::new(HEAD_DIM);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("transpose_heads_source", kernel.source());
    Ok(())
}

#[test]
fn test_merge_heads_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::transpose_heads::MergeHeads::new(HEAD_DIM);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("merge_heads_source", kernel.source());
    Ok(())
}

// ── CausalFlashAttn2 ─────────────────────────────────────────────────────────

#[test]
fn test_causal_flash_attn2_fwd_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teenyformers::kernels::flash_attn2::FlashAttn2CausalFwd::new(HEAD_DIM);
    compile_kernel(&kernel, &sm90_target(), true)?;
    assert_debug_snapshot!("causal_flash_attn2_fwd_source", kernel.source());
    Ok(())
}
