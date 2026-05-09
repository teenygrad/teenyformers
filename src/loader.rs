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

//! Weight loading from HuggingFace safetensors checkpoints.
//!
//! # Fused gate+up projection
//!
//! Our `FeedForward` fuses gate and up projections into a single weight matrix
//! of shape `[d_model, 2*d_ff]`, named `mlp.gate_up_proj.weight` in the graph.
//! HuggingFace Llama checkpoints store them separately as:
//!   - `model.layers.N.mlp.gate_proj.weight`  — shape `[d_ff, d_model]`
//!   - `model.layers.N.mlp.up_proj.weight`    — shape `[d_ff, d_model]`
//!
//! The loader concatenates these along axis 0 → `[2*d_ff, d_model]`.

use std::{
    collections::HashMap,
    path::Path,
};

use anyhow::{Result, anyhow, bail};
use safetensors::Dtype;
use teeny_cuda::model::LoadedModel;
use teeny_data::safetensors::SafeTensors;

/// Load all Llama-format `.safetensors` files from `model_dir` into `model`.
///
/// Handles BF16 and F16 → F32 conversion automatically.
/// Concatenates `gate_proj` + `up_proj` into the fused `gate_up_proj` slot.
pub fn load_llama_weights(model: &mut LoadedModel, model_dir: &Path) -> Result<()> {
    let raw = load_all_tensors(model_dir)?;

    let named: Vec<(String, usize, usize)> = model.param_info_named().collect();

    for (key, node_idx, param_idx) in named {
        let data = if key.ends_with(".gate_up_proj.weight") {
            // Fused: concatenate gate_proj + up_proj along dim 0.
            let gate_key = key.replace(".gate_up_proj.weight", ".gate_proj.weight");
            let up_key   = key.replace(".gate_up_proj.weight", ".up_proj.weight");
            let gate = raw.get(&gate_key).ok_or_else(|| anyhow!("missing: {gate_key}"))?;
            let up   = raw.get(&up_key).ok_or_else(|| anyhow!("missing: {up_key}"))?;
            [gate.as_slice(), up.as_slice()].concat()
        } else {
            raw.get(&key).ok_or_else(|| anyhow!("missing weight: {key}"))?.clone()
        };

        model.load_param_f32(node_idx, param_idx, &data)?;
    }

    Ok(())
}

/// Read every `.safetensors` file in `dir`, decode all tensors to f32,
/// and return a key → data map.
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
            let f32_data = tensor_to_f32(tensor.dtype(), tensor.data())?;
            out.insert(name.to_string(), f32_data);
        }
    }

    Ok(out)
}

fn tensor_to_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            let n = bytes.len() / 4;
            let mut out = vec![0.0_f32; n];
            for (i, chunk) in bytes.chunks_exact(4).enumerate() {
                out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let n = bytes.len() / 2;
            let mut out = vec![0.0_f32; n];
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                // BF16 → F32: zero-extend the 16-bit bfloat16 to 32-bit float
                out[i] = f32::from_bits((bits as u32) << 16);
            }
            Ok(out)
        }
        Dtype::F16 => {
            let n = bytes.len() / 2;
            let mut out = vec![0.0_f32; n];
            for (i, chunk) in bytes.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                out[i] = f16_to_f32(bits);
            }
            Ok(out)
        }
        other => bail!("unsupported tensor dtype: {other:?}"),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign     = ((h & 0x8000) as u32) << 16;
    let exponent = ((h & 0x7C00) as u32) >> 10;
    let mantissa = ((h & 0x03FF) as u32);

    let bits = if exponent == 0 {
        if mantissa == 0 { sign } else {
            // Subnormal — normalize
            let mut exp = 127 - 14;
            let mut mant = mantissa;
            while mant & 0x400 == 0 { mant <<= 1; exp -= 1; }
            sign | ((exp as u32) << 23) | ((mant & 0x3FF) << 13)
        }
    } else if exponent == 0x1F {
        sign | 0x7F80_0000 | (mantissa << 13) // inf / nan
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}
