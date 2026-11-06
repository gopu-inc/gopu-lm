//! Lecture/ecriture de model.safetensors (compatible) et .b8n (format natif gopu, avec CRC32).
//!
//! .b8n : "B8N\0" | version u16 LE | taille en-tete u64 LE | en-tete JSON (meme schema que safetensors)
//!        | donnees des tenseurs (f32 LE) | CRC32 u32 LE de tout ce qui precede.

use crate::json::{self, Value};
use crate::nn::Mlp;
use crate::tensor::Tensor;
use std::fs;
use std::path::Path;

pub type Tensors = Vec<(String, Tensor)>;
pub type Meta = Vec<(String, String)>;

const MAGIC: &[u8; 4] = b"B8N\0";
const B8N_VERSION: u16 = 1;

pub fn crc32(data: &[u8]) -> u32 {
    let mut c: u32 = 0xFFFF_FFFF;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0xEDB8_8320 } else { c >> 1 };
        }
    }
    !c
}

fn build(tensors: &[(String, Tensor)], meta: &[(String, String)]) -> (Vec<u8>, Vec<u8>) {
    let mut data: Vec<u8> = Vec::new();
    let mut entries: Vec<(String, Value)> = Vec::new();
    if !meta.is_empty() {
        let m: Vec<(String, Value)> = meta.iter().map(|(k, v)| (k.clone(), Value::Str(v.clone()))).collect();
        entries.push(("__metadata__".to_string(), Value::Obj(m)));
    }
    for (name, t) in tensors {
        let start = data.len();
        for x in &t.data {
            data.extend_from_slice(&x.to_le_bytes());
        }
        let end = data.len();
        let shape: Vec<Value> = t.shape.iter().map(|&d| Value::Num(d as f64)).collect();
        let entry = Value::Obj(vec![
            ("dtype".to_string(), Value::Str("F32".to_string())),
            ("shape".to_string(), Value::Arr(shape)),
            (
                "data_offsets".to_string(),
                Value::Arr(vec![Value::Num(start as f64), Value::Num(end as f64)]),
            ),
        ]);
        entries.push((name.clone(), entry));
    }
    let mut header = json::to_string(&Value::Obj(entries)).into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    (header, data)
}

fn parse_body(header: &[u8], data: &[u8]) -> Result<(Tensors, Meta), String> {
    let text = std::str::from_utf8(header).map_err(|e| e.to_string())?;
    let v = json::parse(text)?;
    let obj = match v {
        Value::Obj(o) => o,
        _ => return Err("en-tete invalide".to_string()),
    };
    let mut tensors: Tensors = Vec::new();
    let mut meta: Meta = Vec::new();
    for (k, val) in obj {
        if k == "__metadata__" {
            if let Value::Obj(m) = val {
                for (mk, mv) in m {
                    if let Value::Str(s) = mv {
                        meta.push((mk, s));
                    }
                }
            }
            continue;
        }
        let dtype = val.get("dtype").and_then(|d| d.as_str()).ok_or("dtype manquant")?;
        if dtype != "F32" {
            return Err(format!("dtype {} non supporte (F32 uniquement)", dtype));
        }
        let shape: Vec<usize> = val
            .get("shape")
            .and_then(|s| s.as_arr())
            .ok_or("shape manquant")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.0) as usize)
            .collect();
        let offs = val.get("data_offsets").and_then(|s| s.as_arr()).ok_or("data_offsets manquants")?;
        if offs.len() != 2 {
            return Err("data_offsets invalides".to_string());
        }
        let s = offs[0].as_f64().unwrap_or(0.0) as usize;
        let e = offs[1].as_f64().unwrap_or(0.0) as usize;
        if s > e || e > data.len() || (e - s) % 4 != 0 {
            return Err(format!("offsets hors limites pour {}", k));
        }
        let floats: Vec<f32> = data[s..e]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        tensors.push((k, Tensor::from_vec(floats, &shape)?));
    }
    Ok((tensors, meta))
}

fn rd_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_le_bytes(a)
}

pub fn write_safetensors(path: &str, tensors: &[(String, Tensor)], meta: &[(String, String)]) -> Result<(), String> {
    let (header, data) = build(tensors, meta);
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    fs::write(path, out).map_err(|e| e.to_string())
}

pub fn read_safetensors(path: &str) -> Result<(Tensors, Meta), String> {
    let b = fs::read(path).map_err(|e| e.to_string())?;
    if b.len() < 8 {
        return Err("fichier trop court".to_string());
    }
    let hl = rd_u64(&b) as usize;
    if hl > b.len() - 8 {
        return Err("taille d'en-tete invalide".to_string());
    }
    parse_body(&b[8..8 + hl], &b[8 + hl..])
}

pub fn write_b8n(path: &str, tensors: &[(String, Tensor)], meta: &[(String, String)]) -> Result<(), String> {
    let (header, data) = build(tensors, meta);
    let mut out = Vec::with_capacity(18 + header.len() + data.len() + 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&B8N_VERSION.to_le_bytes());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    let crc = crc32(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    fs::write(path, out).map_err(|e| e.to_string())
}

pub fn read_b8n(path: &str) -> Result<(Tensors, Meta), String> {
    let b = fs::read(path).map_err(|e| e.to_string())?;
    if b.len() < 4 + 2 + 8 + 4 {
        return Err("fichier .b8n trop court".to_string());
    }
    if &b[0..4] != MAGIC {
        return Err("signature .b8n invalide".to_string());
    }
    let body_end = b.len() - 4;
    let stored = u32::from_le_bytes([b[body_end], b[body_end + 1], b[body_end + 2], b[body_end + 3]]);
    if crc32(&b[..body_end]) != stored {
        return Err("CRC32 invalide : fichier .b8n corrompu".to_string());
    }
    let version = u16::from_le_bytes([b[4], b[5]]);
    if version != B8N_VERSION {
        return Err(format!("version .b8n {} non supportee", version));
    }
    let hl = rd_u64(&b[6..14]) as usize;
    let hdr_start = 14;
    if hl > body_end - hdr_start {
        return Err("taille d'en-tete invalide".to_string());
    }
    parse_body(&b[hdr_start..hdr_start + hl], &b[hdr_start + hl..body_end])
}

fn ext(path: &str) -> String {
    Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

pub fn save_tensors(path: &str, tensors: &[(String, Tensor)], meta: &[(String, String)]) -> Result<(), String> {
    match ext(path).as_str() {
        "safetensors" => write_safetensors(path, tensors, meta),
        "b8n" => write_b8n(path, tensors, meta),
        other => Err(format!("extension '{}' inconnue (utiliser .safetensors ou .b8n)", other)),
    }
}

pub fn load_tensors(path: &str) -> Result<(Tensors, Meta), String> {
    match ext(path).as_str() {
        "safetensors" => read_safetensors(path),
        "b8n" => read_b8n(path),
        other => Err(format!("extension '{}' inconnue (utiliser .safetensors ou .b8n)", other)),
    }
}

/// Sauvegarde le modele : le format est choisi selon l'extension (.safetensors ou .b8n).
pub fn save_model(model: &Mlp, path: &str) -> Result<(), String> {
    let (t, meta) = model.export();
    save_tensors(path, &t, &meta)
}

pub fn load_model(path: &str) -> Result<Mlp, String> {
    let (t, meta) = load_tensors(path)?;
    Mlp::import(&t, &meta)
}
