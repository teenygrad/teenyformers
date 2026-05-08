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

//! Graph compilation tests for teenyformers blocks.
//!
//! These tests trace a block's computation graph, lower it via TritonLowering,
//! compile every kernel to PTX via LLVM, and verify all PTX files exist on disk.
//!
//! Requirements:
//! - `cuda` feature enabled (default)
//! - `TEENYC_PATH` env var pointing to the teenyc LLVM compiler binary
//! - `TEENYC_CACHE_DIR` (optional, defaults to /tmp/teenyc_cache)
//!
//! No live GPU is needed — only the host-side PTX compiler.

#[cfg(feature = "cuda")]
mod compile_tests {
    use std::path::Path;

    use dotenv::dotenv;
    use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
    use teeny_compiler::compiler::target::cuda::Target;
    use teeny_core::{
        graph::{DtypeRepr, Op, SymTensor},
        model::LoweringMode,
    };
    use teeny_cuda::{compiler::{graph::CudaGraphCompiler, target::Capability}, testing};
    use teeny_kernels::graph::TritonLowering;

    // ── Fixture dims (match generate.py) ──────────────────────────────────────
    const D_MODEL:   usize = 32;
    const N_HEADS:   usize = 4;
    const D_FF:      usize = 48;
    const ROPE_BASE: f32   = 10_000.0;
    const EPS:       f64   = 1e-6;

    fn make_compiler() -> anyhow::Result<CudaGraphCompiler> {
        let teenyc_path = std::env::var("TEENYC_PATH")
            .expect("TEENYC_PATH must be set to run graph compilation tests");
        let cache_dir = std::env::var("TEENYC_CACHE_DIR")
            .unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());
        let compiler = LlvmCompiler::new(teenyc_path, cache_dir)?;
        Ok(CudaGraphCompiler::new(compiler))
    }

    fn compile_target() -> Target {
        // Use a concrete capability. If a live device is available, prefer its
        // capability; otherwise fall back to Sm90.
        dotenv().ok();
        testing::setup_cuda_env()
            .map(|env| Target::new(env.capability))
            .unwrap_or_else(|_| Target::new(Capability::Sm90))
    }

    fn assert_all_ptx_exist(model: &teeny_cuda::model::CudaModel<'_>) -> usize {
        let dag = &model.dag;
        let topo = dag.topological_sort();
        let mut compiled = 0usize;
        for &idx in &topo {
            let cn = &dag.node(idx).value;
            if cn.ptx_path.is_empty() {
                // Input placeholder — expected.
                continue;
            }
            assert!(
                Path::new(&cn.ptx_path).exists(),
                "PTX file missing for node {idx}: {}",
                cn.ptx_path,
            );
            compiled += 1;
        }
        compiled
    }

    // ── EncoderBlock ──────────────────────────────────────────────────────────

    #[test]
    fn test_encoder_block_graph_compiles() -> anyhow::Result<()> {
        dotenv().ok();
        let target = compile_target();

        let (x, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(D_MODEL)]);
        let block = teenyformers::layers::EncoderBlock::new(
            D_MODEL, N_HEADS, D_FF, ROPE_BASE, EPS,
        );
        let _out = block.forward(x);
        let graph = graph.borrow();

        let graph_compiler = make_compiler()?;
        let lowering = TritonLowering::new();
        let model = graph_compiler.compile_model(
            &graph, &lowering, &target, LoweringMode::Inference, false,
        )?;

        let n_compiled = assert_all_ptx_exist(&model);
        println!("[encoder_block] {n_compiled} kernels compiled, {} total nodes", model.dag.len());
        assert!(n_compiled > 0, "expected at least one compiled kernel");
        Ok(())
    }

    // ── DecoderBlock ──────────────────────────────────────────────────────────

    #[test]
    fn test_decoder_block_graph_compiles() -> anyhow::Result<()> {
        dotenv().ok();
        let target = compile_target();

        let (tgt, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(D_MODEL)]);
        let enc = {
            let node_id = graph.borrow_mut().add_node(
                Op::Input, vec![], DtypeRepr::F32, vec![None, Some(D_MODEL)],
            );
            SymTensor { node_id, graph: graph.clone(), dtype: DtypeRepr::F32, shape: vec![None, Some(D_MODEL)] }
        };
        let block = teenyformers::layers::DecoderBlock::new(
            D_MODEL, N_HEADS, D_FF, ROPE_BASE, EPS,
        );
        let _out = block.forward(tgt, enc);
        let graph = graph.borrow();

        let graph_compiler = make_compiler()?;
        let lowering = TritonLowering::new();
        let model = graph_compiler.compile_model(
            &graph, &lowering, &target, LoweringMode::Inference, false,
        )?;

        let n_compiled = assert_all_ptx_exist(&model);
        println!("[decoder_block] {n_compiled} kernels compiled, {} total nodes", model.dag.len());
        assert!(n_compiled > 0, "expected at least one compiled kernel");
        Ok(())
    }

    // ── Full Transformer ──────────────────────────────────────────────────────

    #[test]
    fn test_full_transformer_graph_compiles() -> anyhow::Result<()> {
        dotenv().ok();
        let target = compile_target();

        let cfg = teenyformers::TransformerConfig {
            d_model: D_MODEL,
            n_heads: N_HEADS,
            n_encoder_layers: 2,
            n_decoder_layers: 2,
            vocab_size: 256,
            max_seq_len: 64,
            eps: EPS as f32,
            rope_base: ROPE_BASE,
        };
        let (src_ids, graph) = SymTensor::input(DtypeRepr::I32, vec![None]);
        let tgt_ids = {
            let node_id = graph.borrow_mut().add_node(
                Op::Input, vec![], DtypeRepr::I32, vec![None],
            );
            SymTensor { node_id, graph: graph.clone(), dtype: DtypeRepr::I32, shape: vec![None] }
        };
        let transformer = teenyformers::Transformer::new(&cfg);
        let _logits = transformer.forward(src_ids, tgt_ids);
        let graph = graph.borrow();

        let graph_compiler = make_compiler()?;
        let lowering = TritonLowering::new();
        let model = graph_compiler.compile_model(
            &graph, &lowering, &target, LoweringMode::Inference, false,
        )?;

        let n_compiled = assert_all_ptx_exist(&model);
        println!("[full_transformer] {n_compiled} kernels compiled, {} total nodes", model.dag.len());
        assert!(n_compiled > 0, "expected at least one compiled kernel");
        Ok(())
    }
}
