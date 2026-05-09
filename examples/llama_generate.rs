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

//! Llama autoregressive text generation.
//!
//! # Setup
//!
//! 1. Install a compatible model (full MHA, n_kv_heads == n_heads):
//!
//! ```bash
//! pip install huggingface_hub
//! # SmolLM2-1.7B uses full MHA and has open weights:
//! huggingface-cli download HuggingFaceTB/SmolLM2-1.7B-Instruct \
//!     --local-dir /tmp/smollm2
//! ```
//!
//! 2. Set the teenyc compiler path:
//!
//! ```bash
//! export TEENYC_PATH=/path/to/teenyc
//! export TEENYC_CACHE_DIR=/tmp/teenyc_cache
//! ```
//!
//! 3. Run:
//!
//! ```bash
//! cargo run --example llama_generate --features cuda -- \
//!     --model-dir /tmp/smollm2 \
//!     --prompt "The quick brown fox"
//! ```
//!
//! # Model compatibility
//!
//! This example requires full multi-head attention (n_kv_heads == n_heads).
//! Grouped-query attention is not yet implemented — loading such a model will
//! print a clear error message.

use std::{
    collections::HashMap,
    env,
    path::{Path, PathBuf},
};

use anyhow::{Result, bail};
use clap::Parser;
use serde::Deserialize;
use teeny_compiler::compiler::backend::llvm::compiler::LlvmCompiler;
use teeny_core::{
    graph::{DtypeRepr, SymTensor},
    model::{LoweringMode},
};
use teeny_cuda::{
    compiler::{graph::CudaGraphCompiler, target::capability_from_device_info},
    compiler::target::Target,
    device::context::Cuda,
    model::{LoadedModel, TensorRef},
};
use teeny_kernels::graph::TritonLowering;
use teenyformers::{LlamaConfig, Llama};
use tokenizers::Tokenizer;

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(about = "Autoregressive text generation with a Llama-style model")]
struct Args {
    /// Directory containing config.json, tokenizer.json, and *.safetensors.
    #[arg(long)]
    model_dir: PathBuf,

    /// Prompt text to complete.
    #[arg(long, default_value = "The quick brown fox")]
    prompt: String,

    /// Maximum number of new tokens to generate.
    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    /// Sampling temperature (0.0 = greedy argmax).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// CUDA device index (0 = first GPU).
    #[arg(long, default_value_t = 0)]
    device_idx: usize,
}

// ── HuggingFace config.json ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct HFConfig {
    hidden_size:             usize,
    num_attention_heads:     usize,
    num_hidden_layers:       usize,
    vocab_size:              usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps:            f64,
    #[serde(default = "default_rope_theta")]
    rope_theta:              f64,
    intermediate_size:       Option<usize>,
    num_key_value_heads:     Option<usize>,
}

fn default_max_pos()    -> usize { 4096 }
fn default_rms_eps()    -> f64   { 1e-5 }
fn default_rope_theta() -> f64   { 10_000.0 }

fn load_hf_config(model_dir: &Path) -> Result<LlamaConfig> {
    let text = std::fs::read_to_string(model_dir.join("config.json"))?;
    let hf: HFConfig = serde_json::from_str(&text)?;

    if let Some(n_kv) = hf.num_key_value_heads {
        if n_kv != hf.num_attention_heads {
            bail!(
                "This model uses grouped-query attention \
                 (num_key_value_heads={n_kv}, num_attention_heads={}).\n\
                 GQA is not yet supported. Use a model with equal KV and Q heads,\n\
                 e.g. SmolLM2-1.7B-Instruct.",
                hf.num_attention_heads
            );
        }
    }

    Ok(LlamaConfig {
        d_model:           hf.hidden_size,
        n_heads:           hf.num_attention_heads,
        n_layers:          hf.num_hidden_layers,
        vocab_size:        hf.vocab_size,
        max_seq_len:       hf.max_position_embeddings,
        eps:               hf.rms_norm_eps,
        rope_base:         hf.rope_theta as f32,
        intermediate_size: hf.intermediate_size,
        n_kv_heads:        None,
    })
}

// ── Weight loading ────────────────────────────────────────────────────────────

use safetensors::Dtype;
use teeny_data::safetensors::SafeTensors;

fn load_all_tensors(dir: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    paths.sort();

    if paths.is_empty() {
        bail!("no .safetensors files found in {}", dir.display());
    }

    let mut out: HashMap<String, Vec<f32>> = HashMap::new();
    for path in &paths {
        let st = SafeTensors::from_pretrained(path)?;
        let view = st.tensors()?;
        for (name, tensor) in view.tensors() {
            let data = tensor_to_f32(tensor.dtype(), tensor.data())?;
            out.insert(name.to_string(), data);
        }
    }
    Ok(out)
}

fn tensor_to_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            Ok(bytes.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        Dtype::BF16 => {
            Ok(bytes.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    f32::from_bits((bits as u32) << 16)
                })
                .collect())
        }
        Dtype::F16 => {
            Ok(bytes.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect())
        }
        other => bail!("unsupported dtype: {other:?}"),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp  = ((h & 0x7C00) as u32) >> 10;
    let mant = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign } else {
            let mut e = 127u32 - 14;
            let mut m = mant;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn load_weights(model: &mut LoadedModel, model_dir: &Path) -> Result<()> {
    let raw = load_all_tensors(model_dir)?;
    let named: Vec<(String, usize, usize)> = model.param_info_named().collect();

    for (key, node_idx, param_idx) in named {
        let data = if key.ends_with(".gate_up_proj.weight") {
            // Fused: concat gate_proj + up_proj along row dim.
            let gate_key = key.replace(".gate_up_proj.weight", ".gate_proj.weight");
            let up_key   = key.replace(".gate_up_proj.weight", ".up_proj.weight");
            let gate = raw.get(&gate_key)
                .ok_or_else(|| anyhow::anyhow!("missing: {gate_key}"))?;
            let up   = raw.get(&up_key)
                .ok_or_else(|| anyhow::anyhow!("missing: {up_key}"))?;
            [gate.as_slice(), up.as_slice()].concat()
        } else {
            raw.get(&key)
                .ok_or_else(|| anyhow::anyhow!("missing weight: {key}"))?
                .clone()
        };

        model.load_param_f32(node_idx, param_idx, &data)?;
    }

    Ok(())
}

// ── Sampling ───────────────────────────────────────────────────────────────────

fn greedy_sample(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn temperature_sample(logits: &[f32], temp: f32, rng: &mut u64) -> u32 {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // xorshift64 RNG
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let r = (*rng as f32) / (u64::MAX as f32);

    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum { return i as u32; }
    }
    (probs.len() - 1) as u32
}

// ── Main ───────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    // ── Config ──────────────────────────────────────────────────────────────────
    eprint!("Loading model config... ");
    let cfg = load_hf_config(&args.model_dir)?;
    eprintln!(
        "d_model={} n_heads={} n_layers={} vocab={} d_ff={}",
        cfg.d_model, cfg.n_heads, cfg.n_layers, cfg.vocab_size, cfg.d_ff()
    );

    // ── Tokenizer ───────────────────────────────────────────────────────────────
    eprint!("Loading tokenizer... ");
    let tok_path = args.model_dir.join("tokenizer.json");
    let tok = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
    eprintln!("ok (vocab size: {})", tok.get_vocab_size(true));

    // ── CUDA device ─────────────────────────────────────────────────────────────
    eprint!("Initialising CUDA... ");
    let cuda = Cuda::try_new()?;
    let device_infos = cuda.list_devices()?;
    if device_infos.len() <= args.device_idx {
        bail!("device index {} out of range ({} devices)", args.device_idx, device_infos.len());
    }
    let device = cuda.device(&device_infos[args.device_idx].id)?;
    let capability = capability_from_device_info(&device.info)?;
    eprintln!("{} ({})", device.info.name, capability);

    // ── Graph tracing ───────────────────────────────────────────────────────────
    eprint!("Tracing model graph... ");
    let (ids, graph) = SymTensor::input(DtypeRepr::F32, vec![None]);
    let model = Llama::new(&cfg);
    let _logits = model.forward(ids);
    let graph = graph.borrow();
    eprintln!("{} nodes", graph.nodes.len());

    // ── Compilation ─────────────────────────────────────────────────────────────
    eprint!("Compiling graph to PTX (may take a few minutes on first run)... ");
    let teenyc_path = env::var("TEENYC_PATH")
        .unwrap_or_else(|_| "teenyc".to_string());
    let ptx_cache = env::var("TEENYC_CACHE_DIR")
        .unwrap_or_else(|_| "/tmp/teenyc_cache".to_string());

    let compiler       = LlvmCompiler::new(teenyc_path, ptx_cache)?;
    let graph_compiler = CudaGraphCompiler::new(compiler);
    let target         = Target::new(capability);
    let lowering       = TritonLowering::new();

    let cuda_model = graph_compiler.compile_model(
        &graph, &lowering, &target, LoweringMode::Inference, false,
    )?;
    eprintln!("{} compiled nodes", cuda_model.dag.len());

    // ── Load into GPU memory ────────────────────────────────────────────────────
    eprint!("Loading model into GPU memory... ");
    // batch_size = 1 for inference (resolves None dims for param sizing)
    let mut loaded = cuda_model.load(&device, 1)?;
    eprintln!("ok");

    // ── Weights ─────────────────────────────────────────────────────────────────
    eprint!("Loading weights... ");
    load_weights(&mut loaded, &args.model_dir)?;
    eprintln!("ok");

    // ── Tokenise prompt ─────────────────────────────────────────────────────────
    let enc = tok.encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let mut token_ids: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("Prompt tokens ({} total): {:?}", token_ids.len(), token_ids);

    // EOS token: try common names, fall back to id=2 (Llama1/2 default).
    let vocab = tok.get_vocab(true);
    let eos_id = vocab.get("</s>")
        .or_else(|| vocab.get("<|endoftext|>"))
        .or_else(|| vocab.get("<eos>"))
        .or_else(|| vocab.get("<|eot_id|>"))
        .copied()
        .unwrap_or(2);

    // Print prompt then generate.
    print!("{}", args.prompt);
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut rng: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64)
        .unwrap_or(42);

    // ── Autoregressive generation loop ──────────────────────────────────────────
    for _step in 0..args.max_new_tokens {
        let ids_f32: Vec<f32> = token_ids.iter().map(|&x| x as f32).collect();
        let seq_len = ids_f32.len();

        let input_ref = TensorRef::from_host_f32(&ids_f32, vec![seq_len])?;

        // Forward pass: logits [S, vocab_size]
        let logits_ref = loaded.forward(&device, seq_len, &[input_ref])?;
        let logits_flat = logits_ref.to_host_f32()?;

        // Logits at the last position.
        let vocab_size = cfg.vocab_size;
        let last_logits = &logits_flat[(seq_len - 1) * vocab_size .. seq_len * vocab_size];

        let next_id = if args.temperature == 0.0 {
            greedy_sample(last_logits)
        } else {
            temperature_sample(last_logits, args.temperature, &mut rng)
        };

        if next_id == eos_id { break; }

        let text = tok.decode(&[next_id], true).unwrap_or_default();
        print!("{text}");
        std::io::stdout().flush()?;

        token_ids.push(next_id);
    }

    println!();
    Ok(())
}
