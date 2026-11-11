//! airtel : modele de langage "next-token" de style Llama, en Rust pur (autograd ecrit a la main).
//!
//! Architecture (identique a `LlamaForCausalLM` de Hugging Face, sans biais, attention multi-tetes) :
//!   embeddings -> N x [RMSNorm -> attention causale + RoPE -> residu -> RMSNorm -> MLP SwiGLU -> residu] -> RMSNorm
//!   -> tete LM liee aux embeddings (`tie_word_embeddings`).
//! Pas de table de positions : le contexte (20 000) ne coute aucun parametre.
//!
//! Fonctions : entrainement complet, LoRA (adaptateurs q/k/v/o/gate/up/down, fusion, format PEFT), DPO
//! (optimisation par preferences), generation avec cache cle/valeur, export Hugging Face.

use crate::io;
use crate::json::{self, Value};
use crate::lm::{attn_backward, attn_forward, sample, GenOpts, GenOut, Sample, TrainOpts, TrainReport};
use crate::tensor::{dot, Rng, Tensor};
use crate::tokenizer::{self, PreTok, Tokenizer};
use std::time::Instant;

// ------------------------------------------------------------------ configuration

#[derive(Clone, Debug, PartialEq)]
pub struct AConfig {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ff: usize,
    pub ctx: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

impl AConfig {
    /// La configuration "airtel-2.1m" : ~2,1 M de parametres, contexte 20 000.
    pub fn airtel_2_1m(vocab: usize) -> AConfig {
        AConfig { vocab, d: 128, layers: 8, heads: 4, ff: 344, ctx: 20000, rope_theta: 10000.0, eps: 1e-5 }
    }
    pub fn head_dim(&self) -> usize {
        self.d / self.heads
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.d == 0 || self.layers == 0 || self.heads == 0 || self.ff == 0 || self.vocab < 300 || self.ctx < 2 {
            return Err("configuration invalide".to_string());
        }
        if self.d % self.heads != 0 || (self.d / self.heads) % 2 != 0 {
            return Err("d doit etre divisible par heads et la dimension d'une tete doit etre paire (RoPE)".to_string());
        }
        Ok(())
    }
    pub fn n_params(&self) -> usize {
        self.vocab * self.d + self.layers * (2 * self.d + 4 * self.d * self.d + 3 * self.d * self.ff) + self.d
    }
    pub fn n_tensors(&self) -> usize {
        2 + 9 * self.layers
    }
    /// Noms Hugging Face (LlamaForCausalLM) dans l'ordre interne.
    pub fn hf_names(&self) -> Vec<String> {
        let mut n = vec!["model.embed_tokens.weight".to_string()];
        for l in 0..self.layers {
            for k in [
                "input_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                n.push(format!("model.layers.{}.{}", l, k));
            }
        }
        n.push("model.norm.weight".to_string());
        n
    }
    /// Formes internes : les matrices lineaires sont stockees [entree, sortie] (y = x.W).
    pub fn shapes(&self) -> Vec<Vec<usize>> {
        let (v, d, f) = (self.vocab, self.d, self.ff);
        let mut s = vec![vec![v, d]];
        for _ in 0..self.layers {
            s.extend([vec![d], vec![d, d], vec![d, d], vec![d, d], vec![d, d], vec![d], vec![d, f], vec![d, f], vec![f, d]]);
        }
        s.push(vec![d]);
        s
    }
}

const EMB: usize = 0;
fn li(l: usize, k: usize) -> usize {
    1 + l * 9 + k
}
fn norm_idx(c: &AConfig) -> usize {
    1 + 9 * c.layers
}
fn is_linear(i: usize, c: &AConfig) -> bool {
    i > 0 && i < norm_idx(c) && matches!((i - 1) % 9, 1 | 2 | 3 | 4 | 6 | 7 | 8)
}

pub const LORA_KINDS: [&str; 7] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
const KIND_K: [usize; 7] = [1, 2, 3, 4, 6, 7, 8];

fn transpose(t: &Tensor) -> Tensor {
    let (r, c) = (t.rows(), t.cols());
    let mut o = Tensor::zeros(&[c, r]);
    for i in 0..r {
        for j in 0..c {
            o.data[j * r + i] = t.data[i * c + j];
        }
    }
    o
}

// ------------------------------------------------------------------ LoRA

#[derive(Clone, Debug)]
pub struct LoraCfg {
    pub r: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub targets: Vec<String>,
}

impl Default for LoraCfg {
    fn default() -> Self {
        LoraCfg { r: 16, alpha: 32.0, dropout: 0.05, targets: vec!["q_proj".to_string(), "v_proj".to_string()] }
    }
}

pub struct Lora {
    pub cfg: LoraCfg,
    slots: Vec<Option<usize>>,
    pub keys: Vec<(usize, usize)>, // (couche, type)
    pub a: Vec<Tensor>,            // [entree, r]
    pub b: Vec<Tensor>,            // [r, sortie]
}

fn kind_dims(c: &AConfig, kind: usize) -> (usize, usize) {
    match kind {
        4 | 5 => (c.d, c.ff),
        6 => (c.ff, c.d),
        _ => (c.d, c.d),
    }
}

impl Lora {
    pub fn new(c: &AConfig, cfg: LoraCfg, seed: u64) -> Result<Lora, String> {
        if cfg.r == 0 {
            return Err("rang LoRA nul".to_string());
        }
        for t in &cfg.targets {
            if !LORA_KINDS.contains(&t.as_str()) {
                return Err(format!("module LoRA inconnu '{}' (attendus : {})", t, LORA_KINDS.join(", ")));
            }
        }
        let mut rng = Rng::new(seed);
        let (mut slots, mut keys, mut a, mut b) = (vec![None; c.layers * 7], Vec::new(), Vec::new(), Vec::new());
        for l in 0..c.layers {
            for (k, name) in LORA_KINDS.iter().enumerate() {
                if cfg.targets.iter().any(|t| t == name) {
                    let (din, dout) = kind_dims(c, k);
                    let bound = 1.0 / (din as f32).sqrt();
                    let mut ta = Tensor::zeros(&[din, cfg.r]);
                    ta.data.iter_mut().for_each(|v| *v = (rng.uniform() * 2.0 - 1.0) * bound);
                    slots[l * 7 + k] = Some(keys.len());
                    keys.push((l, k));
                    a.push(ta);
                    b.push(Tensor::zeros(&[cfg.r, dout]));
                }
            }
        }
        Ok(Lora { cfg, slots, keys, a, b })
    }
    pub fn scale(&self) -> f32 {
        self.cfg.alpha / self.cfg.r as f32
    }
    fn slot(&self, l: usize, k: usize) -> Option<usize> {
        self.slots[l * 7 + k]
    }
    pub fn n_params(&self) -> usize {
        self.a.iter().chain(self.b.iter()).map(|t| t.data.len()).sum()
    }
    fn peft_name(l: usize, k: usize, ab: &str) -> String {
        let sub = if k < 4 { "self_attn" } else { "mlp" };
        format!("base_model.model.model.layers.{}.{}.{}.lora_{}.weight", l, sub, LORA_KINDS[k], ab)
    }
    /// Ecrit adapter_model.safetensors + adapter_config.json (format PEFT).
    pub fn save(&self, dir: &str, base: &str) -> Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{} : {}", dir, e))?;
        let mut ts: Vec<(String, Tensor)> = Vec::new();
        for (s, &(l, k)) in self.keys.iter().enumerate() {
            ts.push((Lora::peft_name(l, k, "A"), transpose(&self.a[s])));
            ts.push((Lora::peft_name(l, k, "B"), transpose(&self.b[s])));
        }
        io::save_tensors(&format!("{}/adapter_model.safetensors", dir), &ts, &[("format".to_string(), "pt".to_string())])?;
        let st = |x: &str| Value::Str(x.to_string());
        let cfg = Value::Obj(vec![
            ("peft_type".into(), st("LORA")),
            ("task_type".into(), st("CAUSAL_LM")),
            ("base_model_name_or_path".into(), st(base)),
            ("r".into(), Value::Num(self.cfg.r as f64)),
            ("lora_alpha".into(), Value::Num(self.cfg.alpha as f64)),
            ("lora_dropout".into(), Value::Num(self.cfg.dropout as f64)),
            ("bias".into(), st("none")),
            ("target_modules".into(), Value::Arr(self.cfg.targets.iter().map(|t| st(t)).collect())),
            ("fan_in_fan_out".into(), Value::Bool(false)),
            ("inference_mode".into(), Value::Bool(true)),
            ("use_rslora".into(), Value::Bool(false)),
            ("modules_to_save".into(), Value::Null),
        ]);
        std::fs::write(format!("{}/adapter_config.json", dir), json::to_string(&cfg)).map_err(|e| e.to_string())
    }
    pub fn load(c: &AConfig, dir: &str) -> Result<Lora, String> {
        let txt = std::fs::read_to_string(format!("{}/adapter_config.json", dir)).map_err(|e| format!("adapter_config.json : {}", e))?;
        let v = json::parse(&txt)?;
        let n = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let targets: Vec<String> = v.get("target_modules").and_then(|x| x.as_arr()).map(|a| a.iter().filter_map(|y| y.as_str().map(|s| s.to_string())).collect()).unwrap_or_default();
        let cfg = LoraCfg { r: n("r", 16.0) as usize, alpha: n("lora_alpha", 32.0) as f32, dropout: n("lora_dropout", 0.0) as f32, targets };
        let mut lo = Lora::new(c, cfg, 1)?;
        let (ts, _) = io::load_tensors(&format!("{}/adapter_model.safetensors", dir))?;
        for (s, &(l, k)) in lo.keys.clone().iter().enumerate() {
            let fa = ts.iter().find(|(n, _)| *n == Lora::peft_name(l, k, "A")).ok_or("tenseur lora_A absent")?;
            let fb = ts.iter().find(|(n, _)| *n == Lora::peft_name(l, k, "B")).ok_or("tenseur lora_B absent")?;
            lo.a[s] = transpose(&fa.1);
            lo.b[s] = transpose(&fb.1);
        }
        Ok(lo)
    }
}

// ------------------------------------------------------------------ operations de base

fn add_assign(dst: &mut Tensor, src: &Tensor) {
    for (a, b) in dst.data.iter_mut().zip(src.data.iter()) {
        *a += *b;
    }
}

fn rms_fwd(x: &Tensor, g: &Tensor, eps: f32) -> (Tensor, Vec<f32>) {
    let (t, d) = (x.rows(), x.cols());
    let mut y = Tensor::zeros(&[t, d]);
    let mut r = vec![0.0f32; t];
    for i in 0..t {
        let row = &x.data[i * d..(i + 1) * d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let rr = 1.0 / (ms + eps).sqrt();
        r[i] = rr;
        for j in 0..d {
            y.data[i * d + j] = row[j] * rr * g.data[j];
        }
    }
    (y, r)
}

fn rms_bwd(dy: &Tensor, x: &Tensor, r: &[f32], g: &Tensor) -> (Tensor, Vec<f32>) {
    let (t, d) = (dy.rows(), dy.cols());
    let mut dx = Tensor::zeros(&[t, d]);
    let mut dg = vec![0.0f32; d];
    for i in 0..t {
        let mut s = 0.0f32;
        for j in 0..d {
            s += dy.data[i * d + j] * g.data[j] * x.data[i * d + j];
        }
        let r3 = r[i] * r[i] * r[i];
        for j in 0..d {
            dx.data[i * d + j] = r[i] * g.data[j] * dy.data[i * d + j] - r3 * x.data[i * d + j] * s / d as f32;
            dg[j] += dy.data[i * d + j] * x.data[i * d + j] * r[i];
        }
    }
    (dx, dg)
}

fn rope_tables(positions: std::ops::Range<usize>, hd: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let (mut cs, mut sn) = (Vec::new(), Vec::new());
    for p in positions {
        for i in 0..half {
            let inv = (theta as f64).powf(-(2.0 * i as f64) / hd as f64);
            let a = p as f64 * inv;
            cs.push(a.cos() as f32);
            sn.push(a.sin() as f32);
        }
    }
    (cs, sn)
}

/// RoPE a la maniere de Hugging Face (rotate_half : paires (i, i + hd/2)). `inverse` = rotation inverse (retropropagation).
fn apply_rope(x: &mut Tensor, heads: usize, hd: usize, cs: &[f32], sn: &[f32], inverse: bool) {
    let (t, d) = (x.rows(), x.cols());
    let half = hd / 2;
    for p in 0..t {
        for h in 0..heads {
            for i in 0..half {
                let (c, s) = (cs[p * half + i], if inverse { -sn[p * half + i] } else { sn[p * half + i] });
                let ia = p * d + h * hd + i;
                let ib = ia + half;
                let (a, b) = (x.data[ia], x.data[ib]);
                x.data[ia] = a * c - b * s;
                x.data[ib] = b * c + a * s;
            }
        }
    }
}

// ------------------------------------------------------------------ modele

pub struct Airtel {
    pub cfg: AConfig,
    pub p: Vec<Tensor>,
    pub tok: Tokenizer,
    pub lora: Option<Lora>,
}

struct LCache {
    xd: Option<Tensor>,
    mask: Option<Vec<f32>>,
    xa: Tensor,
}

struct LayerC {
    x: Tensor,
    r1: Vec<f32>,
    h1: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pr: Vec<f32>,
    att: Tensor,
    x2: Tensor,
    r2: Vec<f32>,
    h2: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
    lc: Vec<Option<LCache>>,
}

struct AFwd {
    layers: Vec<LayerC>,
    xf: Tensor,
    rf: Vec<f32>,
    hf: Tensor,
}

/// Gradients : `base` (vide si le modele est fige), `la`/`lb` (vides si pas de LoRA entrainable).
pub struct AGrads {
    base: Vec<Tensor>,
    la: Vec<Tensor>,
    lb: Vec<Tensor>,
}

fn add_grads(a: &mut AGrads, b: &AGrads) {
    for (x, y) in a.base.iter_mut().zip(b.base.iter()).chain(a.la.iter_mut().zip(b.la.iter())).chain(a.lb.iter_mut().zip(b.lb.iter())) {
        add_assign(x, y);
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Full,
    Lora,
}

impl Airtel {
    pub fn new(mut cfg: AConfig, tok: Tokenizer, seed: u64) -> Result<Airtel, String> {
        if cfg.vocab < tok.vocab_size() {
            cfg.vocab = tok.vocab_size();
        }
        cfg.validate()?;
        let mut rng = Rng::new(seed);
        let resid = 1.0 / (2.0 * cfg.layers as f32).sqrt();
        let mut p = Vec::new();
        for (i, shape) in cfg.shapes().iter().enumerate() {
            let mut t = Tensor::zeros(shape);
            if shape.len() == 1 {
                t.data.iter_mut().for_each(|v| *v = 1.0);
            } else {
                let k = if i == 0 { 0 } else { (i - 1) % 9 };
                let std = if i > 0 && (k == 4 || k == 8) { 0.02 * resid } else { 0.02 };
                t.data.iter_mut().for_each(|v| *v = rng.normal() * std);
            }
            p.push(t);
        }
        Ok(Airtel { cfg, p, tok, lora: None })
    }

    pub fn n_params(&self) -> usize {
        self.cfg.n_params()
    }

    // ---- passe avant

    fn lin(&self, x: &Tensor, l: usize, kind: usize, train: bool, rng: &mut Rng) -> (Tensor, Option<LCache>) {
        let mut y = x.matmul(&self.p[li(l, KIND_K[kind])]);
        if let Some(lo) = &self.lora {
            if let Some(s) = lo.slot(l, kind) {
                let pd = lo.cfg.dropout;
                let (xd, mask) = if train && pd > 0.0 {
                    let keep = 1.0 / (1.0 - pd);
                    let m: Vec<f32> = (0..x.data.len()).map(|_| if rng.uniform() < pd { 0.0 } else { keep }).collect();
                    let mut xd = x.clone();
                    xd.data.iter_mut().zip(m.iter()).for_each(|(v, k)| *v *= *k);
                    (Some(xd), Some(m))
                } else {
                    (None, None)
                };
                let xa = xd.as_ref().unwrap_or(x).matmul(&lo.a[s]);
                let delta = xa.matmul(&lo.b[s]);
                let sc = lo.scale();
                y.data.iter_mut().zip(delta.data.iter()).for_each(|(a, b)| *a += sc * b);
                return (y, Some(LCache { xd, mask, xa }));
            }
        }
        (y, None)
    }

    fn lin_bwd(&self, x: &Tensor, l: usize, kind: usize, dy: &Tensor, lc: &Option<LCache>, g: &mut AGrads) -> Tensor {
        let idx = li(l, KIND_K[kind]);
        if !g.base.is_empty() {
            add_assign(&mut g.base[idx], &x.t_matmul(dy));
        }
        let mut dx = dy.matmul_t(&self.p[idx]);
        if let (Some(lo), Some(c)) = (&self.lora, lc) {
            let s = lo.slot(l, kind).unwrap();
            let sc = lo.scale();
            let mut dxa = dy.matmul_t(&lo.b[s]);
            dxa.data.iter_mut().for_each(|v| *v *= sc);
            if !g.la.is_empty() {
                let mut db = c.xa.t_matmul(dy);
                db.data.iter_mut().for_each(|v| *v *= sc);
                add_assign(&mut g.lb[s], &db);
                add_assign(&mut g.la[s], &c.xd.as_ref().unwrap_or(x).t_matmul(&dxa));
            }
            let mut dxd = dxa.matmul_t(&lo.a[s]);
            if let Some(m) = &c.mask {
                dxd.data.iter_mut().zip(m.iter()).for_each(|(v, k)| *v *= *k);
            }
            add_assign(&mut dx, &dxd);
        }
        dx
    }

    fn forward(&self, ids: &[u32], train: bool, rng: &mut Rng) -> AFwd {
        let c = &self.cfg;
        let (t, d, ff, hd) = (ids.len(), c.d, c.ff, c.head_dim());
        let (cs, sn) = rope_tables(0..t, hd, c.rope_theta);
        let mut x = Tensor::zeros(&[t, d]);
        for (i, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(c.vocab - 1);
            x.data[i * d..(i + 1) * d].copy_from_slice(&self.p[EMB].data[id * d..(id + 1) * d]);
        }
        let mut layers = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            let (h1, r1) = rms_fwd(&x, &self.p[li(l, 0)], c.eps);
            let mut lc: Vec<Option<LCache>> = (0..7).map(|_| None).collect();
            let (mut q, c0) = self.lin(&h1, l, 0, train, rng);
            let (mut k, c1) = self.lin(&h1, l, 1, train, rng);
            let (v, c2) = self.lin(&h1, l, 2, train, rng);
            lc[0] = c0;
            lc[1] = c1;
            lc[2] = c2;
            apply_rope(&mut q, c.heads, hd, &cs, &sn, false);
            apply_rope(&mut k, c.heads, hd, &cs, &sn, false);
            let (att, pr) = attn_forward(&q, &k, &v, c.heads);
            let (o, c3) = self.lin(&att, l, 3, train, rng);
            lc[3] = c3;
            let mut x2 = x.clone();
            add_assign(&mut x2, &o);
            let (h2, r2) = rms_fwd(&x2, &self.p[li(l, 5)], c.eps);
            let (gate, c4) = self.lin(&h2, l, 4, train, rng);
            let (up, c5) = self.lin(&h2, l, 5, train, rng);
            lc[4] = c4;
            lc[5] = c5;
            let mut act = Tensor::zeros(&[t, ff]);
            for i in 0..t * ff {
                let g = gate.data[i];
                act.data[i] = g / (1.0 + (-g).exp()) * up.data[i];
            }
            let (m, c6) = self.lin(&act, l, 6, train, rng);
            lc[6] = c6;
            let mut x3 = x2.clone();
            add_assign(&mut x3, &m);
            layers.push(LayerC { x, r1, h1, q, k, v, pr: if train { pr } else { Vec::new() }, att, x2, r2, h2, gate, up, act, lc });
            x = x3;
        }
        let (hf, rf) = rms_fwd(&x, &self.p[norm_idx(c)], c.eps);
        AFwd { layers, xf: x, rf, hf }
    }

    /// Perte (entropie croisee ponderee) et gradient sur les logits. Poids negatifs autorises (DPO).
    /// Retourne (dlogits, somme ponderee des -log p pour poids > 0, somme des poids > 0).
    fn ce_grad(&self, fw: &AFwd, tgt: &[Option<(usize, f32)>]) -> (Tensor, f32, f32) {
        let mut dl = fw.hf.matmul_t(&self.p[EMB]);
        let n = dl.cols();
        let (mut loss, mut ws) = (0.0f32, 0.0f32);
        for (r, t) in tgt.iter().enumerate() {
            let row = &mut dl.data[r * n..(r + 1) * n];
            match t {
                Some((tk, w)) if *w != 0.0 => {
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for x in row.iter_mut() {
                        *x = (*x - mx).exp();
                        sum += *x;
                    }
                    let inv = 1.0 / sum;
                    row.iter_mut().for_each(|x| *x *= inv);
                    if *w > 0.0 {
                        loss += w * -(row[*tk].max(1e-12)).ln();
                        ws += w;
                    }
                    row.iter_mut().for_each(|x| *x *= *w);
                    row[*tk] -= *w;
                }
                _ => row.iter_mut().for_each(|x| *x = 0.0),
            }
        }
        (dl, loss, ws)
    }

    fn backward(&self, fw: &AFwd, ids: &[u32], dl: &Tensor, g: &mut AGrads) {
        let c = &self.cfg;
        let (t, d, ff, hd) = (ids.len(), c.d, c.ff, c.head_dim());
        let (cs, sn) = rope_tables(0..t, hd, c.rope_theta);
        let dhf = dl.matmul(&self.p[EMB]);
        if !g.base.is_empty() {
            add_assign(&mut g.base[EMB], &dl.t_matmul(&fw.hf));
        }
        let (mut dx, dgn) = rms_bwd(&dhf, &fw.xf, &fw.rf, &self.p[norm_idx(c)]);
        if !g.base.is_empty() {
            g.base[norm_idx(c)].data.iter_mut().zip(dgn.iter()).for_each(|(a, b)| *a += *b);
        }
        for l in (0..c.layers).rev() {
            let ch = &fw.layers[l];
            let dact = self.lin_bwd(&ch.act, l, 6, &dx, &ch.lc[6], g);
            let mut dgate = Tensor::zeros(&[t, ff]);
            let mut dup = Tensor::zeros(&[t, ff]);
            for i in 0..t * ff {
                let gt = ch.gate.data[i];
                let sg = 1.0 / (1.0 + (-gt).exp());
                dgate.data[i] = dact.data[i] * ch.up.data[i] * sg * (1.0 + gt * (1.0 - sg));
                dup.data[i] = dact.data[i] * gt * sg;
            }
            let mut dh2 = self.lin_bwd(&ch.h2, l, 4, &dgate, &ch.lc[4], g);
            add_assign(&mut dh2, &self.lin_bwd(&ch.h2, l, 5, &dup, &ch.lc[5], g));
            let (dxn2, dg2) = rms_bwd(&dh2, &ch.x2, &ch.r2, &self.p[li(l, 5)]);
            if !g.base.is_empty() {
                g.base[li(l, 5)].data.iter_mut().zip(dg2.iter()).for_each(|(a, b)| *a += *b);
            }
            let mut dx2 = dx;
            add_assign(&mut dx2, &dxn2);
            let datt = self.lin_bwd(&ch.att, l, 3, &dx2, &ch.lc[3], g);
            let (mut dq, mut dk, dv) = attn_backward(&datt, &ch.q, &ch.k, &ch.v, &ch.pr, c.heads);
            apply_rope(&mut dq, c.heads, hd, &cs, &sn, true);
            apply_rope(&mut dk, c.heads, hd, &cs, &sn, true);
            let mut dh1 = self.lin_bwd(&ch.h1, l, 0, &dq, &ch.lc[0], g);
            add_assign(&mut dh1, &self.lin_bwd(&ch.h1, l, 1, &dk, &ch.lc[1], g));
            add_assign(&mut dh1, &self.lin_bwd(&ch.h1, l, 2, &dv, &ch.lc[2], g));
            let (dxn1, dg1) = rms_bwd(&dh1, &ch.x, &ch.r1, &self.p[li(l, 0)]);
            if !g.base.is_empty() {
                g.base[li(l, 0)].data.iter_mut().zip(dg1.iter()).for_each(|(a, b)| *a += *b);
            }
            dx = dx2;
            add_assign(&mut dx, &dxn1);
        }
        if !g.base.is_empty() {
            for (i, &id) in ids.iter().enumerate() {
                let id = (id as usize).min(c.vocab - 1);
                for j in 0..d {
                    g.base[EMB].data[id * d + j] += dx.data[i * d + j];
                }
            }
        }
    }

    /// Gradient d'une sequence (poids par jeton cible `w[i]`). Retourne (perte, somme des poids).
    pub fn seq_grad(&self, g: &mut AGrads, seq: &[u32], w: &[f32], rng: &mut Rng) -> (f32, f32) {
        let n = seq.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        let t = n - 1;
        let fw = self.forward(&seq[..t], true, rng);
        let tgt: Vec<Option<(usize, f32)>> = (0..t).map(|r| Some((seq[r + 1] as usize, w[r + 1]))).collect();
        let (dl, l, ws) = self.ce_grad(&fw, &tgt);
        self.backward(&fw, &seq[..t], &dl, g);
        (l, ws)
    }

    pub fn grads_full(&self) -> AGrads {
        AGrads { base: self.p.iter().map(|x| Tensor::zeros(&x.shape)).collect(), la: vec![], lb: vec![] }
    }

    pub fn grads_lora(&self) -> AGrads {
        let lo = self.lora.as_ref().expect("LoRA absent");
        AGrads { base: vec![], la: lo.a.iter().map(|x| Tensor::zeros(&x.shape)).collect(), lb: lo.b.iter().map(|x| Tensor::zeros(&x.shape)).collect() }
    }

    /// Somme des log-probabilites des jetons `ids[start..]` (chacun predit a partir de son prefixe).
    pub fn completion_logp(&self, ids: &[u32], start: usize) -> f32 {
        let n = ids.len();
        if n < 2 || start >= n {
            return 0.0;
        }
        let fw = self.forward(&ids[..n - 1], false, &mut Rng::new(0));
        self.logp_from(&fw, ids, start)
    }

    fn logp_from(&self, fw: &AFwd, ids: &[u32], start: usize) -> f32 {
        let n = ids.len();
        let (v, d) = (self.cfg.vocab, self.cfg.d);
        let mut total = 0.0f32;
        for r in start.max(1) - 1..n - 1 {
            let h = &fw.hf.data[r * d..(r + 1) * d];
            let lg: Vec<f32> = (0..v).map(|k| dot(h, &self.p[EMB].data[k * d..(k + 1) * d])).collect();
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + lg.iter().map(|x| (x - mx).exp()).sum::<f32>().ln();
            total += lg[ids[r + 1] as usize] - lse;
        }
        total
    }

    // ---- entrainement (complet ou LoRA)

    pub fn eval_loss(&self, samples: &[Sample]) -> f32 {
        let mut rng = Rng::new(7);
        let (mut l, mut w) = (0.0f32, 0.0f32);
        for s in samples {
            let n = s.ids.len().min(self.cfg.ctx + 1);
            if n < 2 {
                continue;
            }
            let fw = self.forward(&s.ids[..n - 1], false, &mut rng);
            let tgt: Vec<Option<(usize, f32)>> = (0..n - 1).map(|r| Some((s.ids[r + 1] as usize, s.w[r + 1]))).collect();
            let (_, a, b) = self.ce_grad(&fw, &tgt);
            l += a;
            w += b;
        }
        if w > 0.0 {
            l / w
        } else {
            f32::NAN
        }
    }

    pub fn train(&mut self, mode: Mode, train: &[Sample], val: &[Sample], o: &TrainOpts, log: &mut dyn FnMut(&str)) -> Result<TrainReport, String> {
        if train.is_empty() {
            return Err("aucun exemple d'entrainement".to_string());
        }
        if mode == Mode::Lora && self.lora.is_none() {
            return Err("aucun adaptateur LoRA : cree-le d'abord".to_string());
        }
        let vocab = self.cfg.vocab as u32;
        if train.iter().any(|s| s.ids.iter().any(|&i| i >= vocab)) {
            return Err("un jeton depasse le vocabulaire du modele".to_string());
        }
        let ctx = self.cfg.ctx;
        let nthreads = if o.threads > 0 { o.threads } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) }.min(o.batch.max(1));
        let mut rng = Rng::new(o.seed);
        let shapes: Vec<Vec<usize>> = match mode {
            Mode::Full => self.p.iter().map(|x| x.shape.clone()).collect(),
            Mode::Lora => {
                let lo = self.lora.as_ref().unwrap();
                lo.a.iter().chain(lo.b.iter()).map(|x| x.shape.clone()).collect()
            }
        };
        let mut m: Vec<Tensor> = shapes.iter().map(|s| Tensor::zeros(s)).collect();
        let mut v: Vec<Tensor> = shapes.iter().map(|s| Tensor::zeros(s)).collect();
        let start = Instant::now();
        let (mut ema, mut last_val, mut done) = (None::<f32>, None::<f32>, 0usize);
        for step in 1..=o.steps {
            let lr = lr_at(step, o);
            let mut batch: Vec<(&[u32], &[f32])> = Vec::with_capacity(o.batch);
            for _ in 0..o.batch {
                let s = &train[rng.below(train.len())];
                let n = s.ids.len();
                if n <= ctx + 1 {
                    batch.push((&s.ids[..], &s.w[..]));
                } else {
                    let st = rng.below(n - ctx);
                    batch.push((&s.ids[st..st + ctx + 1], &s.w[st..st + ctx + 1]));
                }
            }
            let seeds: Vec<u64> = (0..batch.len()).map(|_| rng.below(1 << 30) as u64 + 1).collect();
            let this = &*self;
            let chunk = (batch.len() + nthreads - 1) / nthreads;
            let results: Vec<(AGrads, f32, f32)> = std::thread::scope(|sc| {
                let hs: Vec<_> = batch
                    .chunks(chunk)
                    .zip(seeds.chunks(chunk))
                    .map(|(part, sd)| {
                        sc.spawn(move || {
                            let mut g = if mode == Mode::Full { this.grads_full() } else { this.grads_lora() };
                            let (mut l, mut w) = (0.0f32, 0.0f32);
                            for ((ids, ws), s) in part.iter().zip(sd.iter()) {
                                let (a, b) = this.seq_grad(&mut g, ids, ws, &mut Rng::new(*s));
                                l += a;
                                w += b;
                            }
                            (g, l, w)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("thread")).collect()
            });
            let mut it = results.into_iter();
            let (mut g, mut loss, mut wsum) = it.next().ok_or("lot vide")?;
            for (gi, li_, wi) in it {
                add_grads(&mut g, &gi);
                loss += li_;
                wsum += wi;
            }
            if wsum <= 0.0 {
                continue;
            }
            let inv = 1.0 / wsum;
            let sq: f32 = g.base.iter().chain(g.la.iter()).chain(g.lb.iter()).map(|t| t.data.iter().map(|x| (x * inv) * (x * inv)).sum::<f32>()).sum();
            let norm = sq.sqrt();
            if !norm.is_finite() {
                return Err(format!("gradient non fini a l'etape {} : baisse lr", step));
            }
            let scale = inv * if o.clip > 0.0 && norm > o.clip { o.clip / norm } else { 1.0 };
            match mode {
                Mode::Full => {
                    let c = self.cfg.clone();
                    for i in 0..self.p.len() {
                        let decay = if self.p[i].shape.len() == 2 && is_linear(i, &c) { o.wd } else { 0.0 };
                        adam(&mut self.p[i], &g.base[i], &mut m[i], &mut v[i], lr, decay, scale, step as i32);
                    }
                }
                Mode::Lora => {
                    let lo = self.lora.as_mut().unwrap();
                    let n = lo.a.len();
                    for s in 0..n {
                        adam(&mut lo.a[s], &g.la[s], &mut m[s], &mut v[s], lr, 0.0, scale, step as i32);
                        adam(&mut lo.b[s], &g.lb[s], &mut m[n + s], &mut v[n + s], lr, 0.0, scale, step as i32);
                    }
                }
            }
            done = step;
            let main = loss / wsum;
            ema = Some(ema.map(|e| 0.95 * e + 0.05 * main).unwrap_or(main));
            if o.eval_every > 0 && !val.is_empty() && (step % o.eval_every == 0 || step == o.steps) {
                last_val = Some(self.eval_loss(val));
            }
            if o.log_every > 0 && (step % o.log_every == 0 || step == 1 || step == o.steps) {
                let vs = last_val.map(|x| format!(" | val {:.3}", x)).unwrap_or_default();
                log(&format!("etape {:>5}/{} | perte {:.3} (lissee {:.3}){} | lr {:.5} | |g| {:.2} | {:.0}s", step, o.steps, main, ema.unwrap_or(main), vs, lr, norm, start.elapsed().as_secs_f32()));
            }
            if o.max_seconds > 0.0 && start.elapsed().as_secs_f32() > o.max_seconds {
                log(&format!("temps maximum atteint : arret a l'etape {}", step));
                break;
            }
        }
        if !val.is_empty() {
            last_val = Some(self.eval_loss(val));
        }
        Ok(TrainReport { steps: done, loss: ema.unwrap_or(f32::NAN), val_loss: last_val, seconds: start.elapsed().as_secs_f32() })
    }

    // ---- LoRA : creation / fusion

    pub fn add_lora(&mut self, cfg: LoraCfg, seed: u64) -> Result<(), String> {
        if self.lora.is_some() {
            return Err("un adaptateur LoRA est deja actif : fusionne-le d'abord (merge_lora)".to_string());
        }
        self.lora = Some(Lora::new(&self.cfg, cfg, seed)?);
        Ok(())
    }

    /// W <- W + (alpha/r) * A.B puis retire l'adaptateur. Le modele resultant est un modele ordinaire.
    pub fn merge_lora(&mut self) -> Result<(), String> {
        let lo = self.lora.take().ok_or("aucun adaptateur LoRA")?;
        let sc = lo.scale();
        for (s, &(l, k)) in lo.keys.iter().enumerate() {
            let delta = lo.a[s].matmul(&lo.b[s]);
            let w = &mut self.p[li(l, KIND_K[k])];
            w.data.iter_mut().zip(delta.data.iter()).for_each(|(a, b)| *a += sc * b);
        }
        Ok(())
    }

    // ---- DPO

    /// Optimisation par preferences (DPO) avec un adaptateur LoRA. Reference = modele au debut de l'appel.
    pub fn dpo(&mut self, pairs: &[Pair], o: &DpoOpts, log: &mut dyn FnMut(&str)) -> Result<DpoReport, String> {
        if pairs.is_empty() {
            return Err("aucune paire de preferences".to_string());
        }
        if self.lora.is_none() {
            self.add_lora(o.lora.clone(), o.seed)?;
        }
        let ctx = self.cfg.ctx;
        let seqs = |p: &Pair| -> (Vec<u32>, Vec<u32>) {
            let mut w = p.prompt.clone();
            w.extend_from_slice(&p.chosen);
            let mut l = p.prompt.clone();
            l.extend_from_slice(&p.rejected);
            (w.into_iter().take(ctx + 1).collect(), l.into_iter().take(ctx + 1).collect())
        };
        // log-probabilites de reference (l'adaptateur vient d'etre cree avec B = 0 : politique = reference)
        let refs: Vec<(f32, f32)> = pairs
            .iter()
            .map(|p| {
                let (w, l) = seqs(p);
                (self.completion_logp(&w, p.prompt.len()), self.completion_logp(&l, p.prompt.len()))
            })
            .collect();
        let n = self.lora.as_ref().unwrap().a.len();
        let mut m: Vec<Tensor> = Vec::new();
        for t in self.lora.as_ref().unwrap().a.iter().chain(self.lora.as_ref().unwrap().b.iter()) {
            m.push(Tensor::zeros(&t.shape));
        }
        let mut v: Vec<Tensor> = m.iter().map(|t| Tensor::zeros(&t.shape)).collect();
        let steps = ((o.epochs * pairs.len() as f32) / o.batch as f32).ceil().max(1.0) as usize;
        let mut rng = Rng::new(o.seed);
        let nthreads = if o.threads > 0 { o.threads } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) }.min(o.batch.max(1));
        let top = TrainOpts { steps, lr: o.lr, warmup: (steps / 10).max(1), min_lr_frac: 0.1, ..TrainOpts::default() };
        let start = Instant::now();
        let (mut acc_ema, mut marg_ema, mut loss_ema) = (0.5f32, 0.0f32, std::f32::consts::LN_2);
        for step in 1..=steps {
            let lr = lr_at(step, &top);
            let idx: Vec<usize> = (0..o.batch).map(|_| rng.below(pairs.len())).collect();
            let seeds: Vec<u64> = (0..o.batch).map(|_| rng.below(1 << 30) as u64 + 1).collect();
            let this = &*self;
            let chunk = (idx.len() + nthreads - 1) / nthreads;
            let results: Vec<(AGrads, f32, f32, f32)> = std::thread::scope(|sc| {
                let hs: Vec<_> = idx
                    .chunks(chunk)
                    .zip(seeds.chunks(chunk))
                    .map(|(part, sd)| {
                        let (refs, seqs) = (&refs, &seqs);
                        sc.spawn(move || {
                            let mut g = this.grads_lora();
                            let (mut l, mut a, mut mg) = (0.0f32, 0.0f32, 0.0f32);
                            for (&i, &s) in part.iter().zip(sd.iter()) {
                                let (w, lz) = seqs(&pairs[i]);
                                let (a1, b1, c1) = this.dpo_pair_grad(&mut g, &w, &lz, pairs[i].prompt.len(), refs[i], o.beta, o.sft_alpha, &mut Rng::new(s));
                                l += a1;
                                a += b1;
                                mg += c1;
                            }
                            (g, l, a, mg)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("thread")).collect()
            });
            let mut it = results.into_iter();
            let (mut g, mut l, mut a, mut mg) = it.next().ok_or("lot vide")?;
            for (gi, l2, a2, m2) in it {
                add_grads(&mut g, &gi);
                l += l2;
                a += a2;
                mg += m2;
            }
            let inv = 1.0 / idx.len() as f32;
            let sq: f32 = g.la.iter().chain(g.lb.iter()).map(|t| t.data.iter().map(|x| (x * inv) * (x * inv)).sum::<f32>()).sum();
            let norm = sq.sqrt();
            if !norm.is_finite() {
                return Err("gradient non fini pendant le DPO : baisse lr ou beta".to_string());
            }
            let scale = inv * if norm > 1.0 { 1.0 / norm } else { 1.0 };
            let lo = self.lora.as_mut().unwrap();
            for s in 0..n {
                adam(&mut lo.a[s], &g.la[s], &mut m[s], &mut v[s], lr, 0.0, scale, step as i32);
                adam(&mut lo.b[s], &g.lb[s], &mut m[n + s], &mut v[n + s], lr, 0.0, scale, step as i32);
            }
            loss_ema = 0.9 * loss_ema + 0.1 * l * inv;
            acc_ema = 0.9 * acc_ema + 0.1 * a * inv;
            marg_ema = 0.9 * marg_ema + 0.1 * mg * inv;
            if step == 1 || step % o.log_every.max(1) == 0 || step == steps {
                log(&format!("dpo {:>4}/{} | perte {:.3} | precision des preferences {:.0}% | marge {:+.2} | lr {:.5} | {:.0}s", step, steps, loss_ema, acc_ema * 100.0, marg_ema, lr, start.elapsed().as_secs_f32()));
            }
        }
        // evaluation finale sur toutes les paires
        let (mut acc, mut marg) = (0.0f32, 0.0f32);
        for (i, p) in pairs.iter().enumerate() {
            let (w, l) = seqs(p);
            let z = (self.completion_logp(&w, p.prompt.len()) - refs[i].0) - (self.completion_logp(&l, p.prompt.len()) - refs[i].1);
            acc += (z > 0.0) as u8 as f32;
            marg += z;
        }
        Ok(DpoReport { steps, accuracy: acc / pairs.len() as f32, margin: marg / pairs.len() as f32, seconds: start.elapsed().as_secs_f32() })
    }

    /// Gradient DPO d'une paire. Retourne (perte, bien classee 0/1, marge).
    #[allow(clippy::too_many_arguments)]
    fn dpo_pair_grad(&self, g: &mut AGrads, w: &[u32], l: &[u32], plen: usize, r: (f32, f32), beta: f32, alpha: f32, rng: &mut Rng) -> (f32, f32, f32) {
        let (nw, nl) = (w.len(), l.len());
        if nw < 2 || nl < 2 {
            return (0.0, 0.0, 0.0);
        }
        let fw_w = self.forward(&w[..nw - 1], true, rng);
        let fw_l = self.forward(&l[..nl - 1], true, rng);
        let lp_w = self.logp_from(&fw_w, w, plen);
        let lp_l = self.logp_from(&fw_l, l, plen);
        let z = beta * ((lp_w - r.0) - (lp_l - r.1));
        let sig_neg = 1.0 / (1.0 + z.exp()); // sigma(-z)
        let loss = if z > 30.0 { 0.0 } else if z < -30.0 { -z } else { (1.0 + (-z).exp()).ln() };
        let coef = beta * sig_neg;
        let ncw = (nw - plen).max(1) as f32;
        let tw: Vec<Option<(usize, f32)>> = (0..nw - 1).map(|i| if i + 1 >= plen { Some((w[i + 1] as usize, coef + alpha / ncw)) } else { None }).collect();
        let tl: Vec<Option<(usize, f32)>> = (0..nl - 1).map(|i| if i + 1 >= plen { Some((l[i + 1] as usize, -coef)) } else { None }).collect();
        let (dlw, _, _) = self.ce_grad(&fw_w, &tw);
        self.backward(&fw_w, &w[..nw - 1], &dlw, g);
        let (dll, _, _) = self.ce_grad(&fw_l, &tl);
        self.backward(&fw_l, &l[..nl - 1], &dll, g);
        (loss, (z > 0.0) as u8 as f32, z / beta)
    }

    // ---- generation avec cache cle/valeur

    fn step(&self, kv: &mut Kv, tok: u32, want_logits: bool) -> Option<Vec<f32>> {
        let c = &self.cfg;
        let (d, ff, hd, heads) = (c.d, c.ff, c.head_dim(), c.heads);
        let pos = kv.len;
        let (cs, sn) = rope_tables(pos..pos + 1, hd, c.rope_theta);
        let mut x = Tensor::zeros(&[1, d]);
        let id = (tok as usize).min(c.vocab - 1);
        x.data.copy_from_slice(&self.p[EMB].data[id * d..(id + 1) * d]);
        let mut rng = Rng::new(0);
        let scale = 1.0 / (hd as f32).sqrt();
        for l in 0..c.layers {
            let (h1, _) = rms_fwd(&x, &self.p[li(l, 0)], c.eps);
            let (mut q, _) = self.lin(&h1, l, 0, false, &mut rng);
            let (mut k, _) = self.lin(&h1, l, 1, false, &mut rng);
            let (v, _) = self.lin(&h1, l, 2, false, &mut rng);
            apply_rope(&mut q, heads, hd, &cs, &sn, false);
            apply_rope(&mut k, heads, hd, &cs, &sn, false);
            kv.k[l].extend_from_slice(&k.data);
            kv.v[l].extend_from_slice(&v.data);
            let mut att = Tensor::zeros(&[1, d]);
            for h in 0..heads {
                let off = h * hd;
                let qh = &q.data[off..off + hd];
                let mut sc: Vec<f32> = (0..=pos).map(|j| dot(qh, &kv.k[l][j * d + off..j * d + off + hd]) * scale).collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in sc.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for (j, s) in sc.iter().enumerate() {
                    let wgt = s / sum;
                    let vj = &kv.v[l][j * d + off..j * d + off + hd];
                    for e in 0..hd {
                        att.data[off + e] += wgt * vj[e];
                    }
                }
            }
            let (o, _) = self.lin(&att, l, 3, false, &mut rng);
            add_assign(&mut x, &o);
            let (h2, _) = rms_fwd(&x, &self.p[li(l, 5)], c.eps);
            let (gate, _) = self.lin(&h2, l, 4, false, &mut rng);
            let (up, _) = self.lin(&h2, l, 5, false, &mut rng);
            let mut act = Tensor::zeros(&[1, ff]);
            for i in 0..ff {
                act.data[i] = gate.data[i] / (1.0 + (-gate.data[i]).exp()) * up.data[i];
            }
            let (m, _) = self.lin(&act, l, 6, false, &mut rng);
            add_assign(&mut x, &m);
        }
        kv.len += 1;
        if !want_logits {
            return None;
        }
        let (hf, _) = rms_fwd(&x, &self.p[norm_idx(c)], c.eps);
        Some((0..c.vocab).map(|v| dot(&hf.data, &self.p[EMB].data[v * d..(v + 1) * d])).collect())
    }

    fn new_kv(&self) -> Kv {
        Kv { k: vec![Vec::new(); self.cfg.layers], v: vec![Vec::new(); self.cfg.layers], len: 0 }
    }

    /// Logits du jeton suivant pour un contexte complet (utilise le cache pour le prefixe).
    pub fn last_logits(&self, ids: &[u32]) -> Vec<f32> {
        let ids = if ids.len() >= self.cfg.ctx { &ids[ids.len() - (self.cfg.ctx - 1)..] } else { ids };
        let mut kv = self.new_kv();
        let mut out = Vec::new();
        for (i, &t) in ids.iter().enumerate() {
            if let Some(l) = self.step(&mut kv, t, i + 1 == ids.len()) {
                out = l;
            }
        }
        out
    }

    pub fn generate(&self, prompt: &[u32], o: &GenOpts) -> GenOut {
        let valid = self.tok.vocab_size();
        let mut seed = o.seed;
        if seed == 0 {
            seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        }
        let mut rng = Rng::new(seed);
        let prompt: &[u32] = if prompt.len() >= self.cfg.ctx { &prompt[prompt.len() - (self.cfg.ctx - 1)..] } else { prompt };
        let mut kv = self.new_kv();
        let mut logits = Vec::new();
        for (i, &t) in prompt.iter().enumerate() {
            if let Some(l) = self.step(&mut kv, t, i + 1 == prompt.len()) {
                logits = l;
            }
        }
        let mut all: Vec<u32> = prompt.to_vec();
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..o.max_tokens {
            if logits.is_empty() || kv.len >= self.cfg.ctx {
                return GenOut { tokens: out, finish: "length", stop: None };
            }
            for x in logits[valid..].iter_mut() {
                *x = f32::NEG_INFINITY;
            }
            let from = all.len().saturating_sub(64);
            let next = sample(&mut logits, o, &mut rng, &all[from..]);
            all.push(next);
            out.push(next);
            if o.stop_ids.contains(&next) {
                return GenOut { tokens: out, finish: "stop_token", stop: Some(self.tok.decode(&[next])) };
            }
            if !o.stop_strings.is_empty() {
                let text = self.tok.decode(&out);
                if let Some(s) = o.stop_strings.iter().find(|s| !s.is_empty() && text.contains(s.as_str())) {
                    return GenOut { tokens: out, finish: "stop_string", stop: Some(s.clone()) };
                }
            }
            logits = self.step(&mut kv, next, true).unwrap_or_default();
        }
        GenOut { tokens: out, finish: "length", stop: None }
    }

    pub fn predict_next(&self, ids: &[u32], top_n: usize) -> Vec<(u32, f32)> {
        if ids.is_empty() {
            return Vec::new();
        }
        let mut lg = self.last_logits(ids);
        for x in lg[self.tok.vocab_size()..].iter_mut() {
            *x = f32::NEG_INFINITY;
        }
        lg[tokenizer::PAD as usize] = f32::NEG_INFINITY;
        crate::lm::top_probs(&lg, top_n)
    }

    pub fn token_embeddings(&self, ids: &[u32]) -> Tensor {
        let d = self.cfg.d;
        let mut t = Tensor::zeros(&[ids.len(), d]);
        for (i, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(self.cfg.vocab - 1);
            t.data[i * d..(i + 1) * d].copy_from_slice(&self.p[EMB].data[id * d..(id + 1) * d]);
        }
        t
    }

    pub fn hidden_states(&self, ids: &[u32]) -> Tensor {
        let ids = if ids.len() > self.cfg.ctx { &ids[ids.len() - self.cfg.ctx..] } else { ids };
        if ids.is_empty() {
            return Tensor::zeros(&[0, self.cfg.d]);
        }
        self.forward(ids, false, &mut Rng::new(0)).hf
    }

    // ---- sauvegarde / chargement / export Hugging Face

    fn hf_tensors(&self) -> Vec<(String, Tensor)> {
        self.cfg.hf_names().into_iter().zip(self.p.iter()).enumerate().map(|(i, (n, t))| (n, if is_linear(i, &self.cfg) { transpose(t) } else { t.clone() })).collect()
    }

    fn meta(&self) -> Vec<(String, String)> {
        let c = &self.cfg;
        vec![
            ("arch".into(), "airtel-llama".into()),
            ("vocab".into(), c.vocab.to_string()),
            ("d".into(), c.d.to_string()),
            ("layers".into(), c.layers.to_string()),
            ("heads".into(), c.heads.to_string()),
            ("ff".into(), c.ff.to_string()),
            ("ctx".into(), c.ctx.to_string()),
            ("rope_theta".into(), c.rope_theta.to_string()),
            ("eps".into(), c.eps.to_string()),
            ("n_specials".into(), tokenizer::SPECIALS.len().to_string()),
            ("tok_pretok".into(), self.tok.pretok().name().into()),
            ("tok_merges".into(), self.tok.merges_to_string()),
            ("framework".into(), "gopu-lm 0.3.0".into()),
        ]
    }

    /// Sauvegarde native .b8n (ou .safetensors) : tenseurs au format Hugging Face + metadonnees.
    pub fn save(&self, path: &str) -> Result<(), String> {
        if self.lora.is_some() {
            return Err("un adaptateur LoRA est actif : appelle merge_lora() (ou sauvegarde l'adaptateur a part)".to_string());
        }
        io::save_tensors(path, &self.hf_tensors(), &self.meta())
    }

    fn from_tensors(ts: Vec<(String, Tensor)>, cfg: AConfig, tok: Tokenizer) -> Result<Airtel, String> {
        cfg.validate()?;
        if tok.vocab_size() > cfg.vocab {
            return Err("le tokenizer depasse le vocabulaire du modele".to_string());
        }
        let mut p = Vec::new();
        for (i, (name, shape)) in cfg.hf_names().iter().zip(cfg.shapes().iter()).enumerate() {
            let (_, t) = ts.iter().find(|(n, _)| n == name).ok_or(format!("tenseur '{}' absent", name))?;
            let t = if is_linear(i, &cfg) { transpose(t) } else { t.clone() };
            if &t.shape != shape {
                return Err(format!("forme inattendue pour '{}' : {:?} au lieu de {:?}", name, t.shape, shape));
            }
            p.push(t);
        }
        Ok(Airtel { cfg, p, tok, lora: None })
    }

    /// Charge un fichier .b8n, un dossier (model.b8n, sinon model.safetensors + config.json + vocab.json + merges.txt)
    /// ou un .safetensors accompagne de ces fichiers.
    pub fn load(path: &str) -> Result<Airtel, String> {
        let p = std::path::Path::new(path);
        if p.is_dir() {
            let b = p.join("model.b8n");
            if b.exists() {
                return Airtel::load(b.to_str().unwrap());
            }
            return Airtel::from_hf_dir(path, &p.join("model.safetensors"));
        }
        if path.ends_with(".safetensors") {
            let dir = p.parent().map(|d| d.to_str().unwrap_or(".")).filter(|d| !d.is_empty()).unwrap_or(".");
            return Airtel::from_hf_dir(dir, p);
        }
        let (ts, meta) = io::load_tensors(path)?;
        let find = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.clone());
        if find("arch").as_deref() != Some("airtel-llama") {
            return Err(format!("architecture '{}' : ce n'est pas un modele airtel", find("arch").unwrap_or_default()));
        }
        let num = |k: &str| -> Result<usize, String> { find(k).ok_or(format!("metadonnee '{}' absente", k))?.parse::<usize>().map_err(|e| format!("'{}' : {}", k, e)) };
        let flt = |k: &str| -> Result<f32, String> { find(k).ok_or(format!("metadonnee '{}' absente", k))?.parse::<f32>().map_err(|e| format!("'{}' : {}", k, e)) };
        let cfg = AConfig { vocab: num("vocab")?, d: num("d")?, layers: num("layers")?, heads: num("heads")?, ff: num("ff")?, ctx: num("ctx")?, rope_theta: flt("rope_theta")?, eps: flt("eps")? };
        if num("n_specials")? != tokenizer::SPECIALS.len() {
            return Err("jeu de jetons speciaux incompatible".to_string());
        }
        let merges = parse_merges(&find("tok_merges").unwrap_or_default())?;
        let tok = Tokenizer::from_merges_mode(merges, PreTok::from_name(&find("tok_pretok").unwrap_or_default()))?;
        Airtel::from_tensors(ts, cfg, tok)
    }

    fn from_hf_dir(dir: &str, st: &std::path::Path) -> Result<Airtel, String> {
        let rd = |f: &str| std::fs::read_to_string(format!("{}/{}", dir, f)).map_err(|e| format!("{}/{} : {}", dir, f, e));
        let cj = json::parse(&rd("config.json")?)?;
        let n = |k: &str| cj.get(k).and_then(|x| x.as_f64()).ok_or(format!("config.json : '{}' absent", k));
        let cfg = AConfig {
            vocab: n("vocab_size")? as usize,
            d: n("hidden_size")? as usize,
            layers: n("num_hidden_layers")? as usize,
            heads: n("num_attention_heads")? as usize,
            ff: n("intermediate_size")? as usize,
            ctx: n("max_position_embeddings")? as usize,
            rope_theta: cj.get("rope_theta").and_then(|x| x.as_f64()).unwrap_or(10000.0) as f32,
            eps: cj.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
        };
        let vj = json::parse(&rd("vocab.json")?)?;
        let mut by_str: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
        if let Value::Obj(o) = &vj {
            for (k, v) in o {
                by_str.insert(k.clone(), v.as_f64().unwrap_or(0.0) as u32);
            }
        }
        let mut merges: Vec<(u32, u32)> = Vec::new();
        for line in rd("merges.txt")?.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let mut it = line.split(' ');
            let (a, b) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
            merges.push((*by_str.get(a).ok_or("fusion inconnue dans vocab.json")?, *by_str.get(b).ok_or("fusion inconnue dans vocab.json")?));
        }
        let pre = if rd("tokenizer.json").map(|t| t.contains("\"Split\"")).unwrap_or(false) { PreTok::Gpt } else { PreTok::Legacy };
        let tok = Tokenizer::from_merges_mode(merges, pre)?;
        let (ts, _) = io::load_tensors(st.to_str().ok_or("chemin invalide")?)?;
        Airtel::from_tensors(ts, cfg, tok)
    }

    /// Ecrit un dossier de modele compatible Hugging Face (LlamaForCausalLM) + le format natif .b8n.
    pub fn export_dir(&self, dir: &str, name: &str) -> Result<(), String> {
        if self.lora.is_some() {
            return Err("fusionne l'adaptateur LoRA avant l'export (merge_lora)".to_string());
        }
        std::fs::create_dir_all(dir).map_err(|e| format!("{} : {}", dir, e))?;
        let w = |f: &str, s: String| std::fs::write(format!("{}/{}", dir, f), s).map_err(|e| format!("{} : {}", f, e));
        io::save_tensors(&format!("{}/model.safetensors", dir), &self.hf_tensors(), &[("format".to_string(), "pt".to_string())])?;
        io::save_tensors(&format!("{}/model.b8n", dir), &self.hf_tensors(), &self.meta())?;
        let c = &self.cfg;
        let st = |x: &str| Value::Str(x.to_string());
        let num = |x: usize| Value::Num(x as f64);
        let cfgj = Value::Obj(vec![
            ("architectures".into(), Value::Arr(vec![st("LlamaForCausalLM")])),
            ("model_type".into(), st("llama")),
            ("vocab_size".into(), num(c.vocab)),
            ("hidden_size".into(), num(c.d)),
            ("intermediate_size".into(), num(c.ff)),
            ("num_hidden_layers".into(), num(c.layers)),
            ("num_attention_heads".into(), num(c.heads)),
            ("num_key_value_heads".into(), num(c.heads)),
            ("head_dim".into(), num(c.head_dim())),
            ("hidden_act".into(), st("silu")),
            ("max_position_embeddings".into(), num(c.ctx)),
            ("initializer_range".into(), Value::Num(0.02)),
            ("rms_norm_eps".into(), Value::Num(c.eps as f64)),
            ("use_cache".into(), Value::Bool(true)),
            ("tie_word_embeddings".into(), Value::Bool(true)),
            ("rope_theta".into(), Value::Num(c.rope_theta as f64)),
            ("rope_scaling".into(), Value::Null),
            ("attention_bias".into(), Value::Bool(false)),
            ("mlp_bias".into(), Value::Bool(false)),
            ("pretraining_tp".into(), num(1)),
            ("bos_token_id".into(), Value::Null),
            ("eos_token_id".into(), num(tokenizer::EOS as usize)),
            ("pad_token_id".into(), num(tokenizer::PAD as usize)),
            ("torch_dtype".into(), st("float32")),
            ("_name_or_path".into(), st(name)),
        ]);
        w("config.json", json::to_string(&cfgj))?;
        let gen = Value::Obj(vec![
            ("_from_model_config".into(), Value::Bool(true)),
            ("eos_token_id".into(), num(tokenizer::EOS as usize)),
            ("pad_token_id".into(), num(tokenizer::PAD as usize)),
            ("do_sample".into(), Value::Bool(true)),
            ("temperature".into(), Value::Num(0.7)),
            ("top_k".into(), num(40)),
            ("top_p".into(), Value::Num(0.95)),
            ("max_new_tokens".into(), num(160)),
        ]);
        w("generation_config.json", json::to_string(&gen))?;
        w("vocab.json", self.tok.hf_vocab_json())?;
        w("merges.txt", self.tok.hf_merges_txt())?;
        w("tokenizer.json", self.tok.hf_tokenizer_json())?;
        let template = "{% for m in messages %}{% if m['role'] == 'system' %}<|system|>\\n{{ m['content'] }}\\n</|system|>\\n{% elif m['role'] == 'user' %}<|user|>\\n{{ m['content'] }}\\n</|user|>\\n{% elif m['role'] == 'assistant' %}<|assistant|>\\n{{ m['content'] }}\\n</|assistant|>\\n{% endif %}{% endfor %}";
        let tc = Value::Obj(vec![
            ("tokenizer_class".into(), st("PreTrainedTokenizerFast")),
            ("model_max_length".into(), num(c.ctx)),
            ("eos_token".into(), st("<|eos|>")),
            ("pad_token".into(), st("<|pad|>")),
            ("bos_token".into(), Value::Null),
            ("unk_token".into(), Value::Null),
            ("clean_up_tokenization_spaces".into(), Value::Bool(false)),
            ("chat_template".into(), Value::Str(template.replace("\\n", "\n"))),
        ]);
        w("tokenizer_config.json", json::to_string(&tc))?;
        let stm = Value::Obj(vec![
            ("eos_token".into(), st("<|eos|>")),
            ("pad_token".into(), st("<|pad|>")),
            ("additional_special_tokens".into(), Value::Arr(tokenizer::SPECIALS.iter().filter(|s| **s != "<|eos|>" && **s != "<|pad|>").map(|s| st(s)).collect())),
        ]);
        w("special_tokens_map.json", json::to_string(&stm))
    }
}

fn parse_merges(s: &str) -> Result<Vec<(u32, u32)>, String> {
    let mut merges = Vec::new();
    for part in s.split(';').filter(|p| !p.is_empty()) {
        let mut it = part.split(',');
        let a = it.next().and_then(|x| x.parse::<u32>().ok()).ok_or("fusion illisible")?;
        let b = it.next().and_then(|x| x.parse::<u32>().ok()).ok_or("fusion illisible")?;
        merges.push((a, b));
    }
    Ok(merges)
}

// ------------------------------------------------------------------ cache KV, optimiseur, types DPO

pub struct Kv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

fn lr_at(step: usize, o: &TrainOpts) -> f32 {
    if step <= o.warmup {
        o.lr * step as f32 / o.warmup.max(1) as f32
    } else {
        let prog = (step - o.warmup) as f32 / (o.steps - o.warmup).max(1) as f32;
        let cos = 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
        o.lr * (o.min_lr_frac + (1.0 - o.min_lr_frac) * cos)
    }
}

#[allow(clippy::too_many_arguments)]
fn adam(p: &mut Tensor, g: &Tensor, m: &mut Tensor, v: &mut Tensor, lr: f32, wd: f32, scale: f32, t: i32) {
    let (b1, b2, eps) = (0.9f32, 0.95f32, 1e-8f32);
    let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
    for k in 0..p.data.len() {
        let gk = g.data[k] * scale;
        m.data[k] = b1 * m.data[k] + (1.0 - b1) * gk;
        v.data[k] = b2 * v.data[k] + (1.0 - b2) * gk * gk;
        let step = (m.data[k] / c1) / ((v.data[k] / c2).sqrt() + eps);
        p.data[k] -= lr * (step + wd * p.data[k]);
    }
}

/// Paire de preference : `prompt` (jetons) puis deux suites : `chosen` (preferee) et `rejected`.
#[derive(Clone, Debug)]
pub struct Pair {
    pub prompt: Vec<u32>,
    pub chosen: Vec<u32>,
    pub rejected: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct DpoOpts {
    pub beta: f32,
    pub epochs: f32,
    pub batch: usize,
    pub lr: f32,
    /// poids d'un terme d'apprentissage supervise sur la reponse preferee (stabilise le DPO)
    pub sft_alpha: f32,
    pub lora: LoraCfg,
    pub seed: u64,
    pub log_every: usize,
    pub threads: usize,
}

impl Default for DpoOpts {
    fn default() -> Self {
        DpoOpts { beta: 0.3, epochs: 2.0, batch: 8, lr: 1e-4, sft_alpha: 0.5, lora: LoraCfg::default(), seed: 1, log_every: 10, threads: 0 }
    }
}

pub struct DpoReport {
    pub steps: usize,
    pub accuracy: f32,
    pub margin: f32,
    pub seconds: f32,
}

/// Lit un fichier JSONL : {"prompt": "...", "chosen": "...", "rejected": "..."} (+ "system" facultatif).
/// `prompt` est le texte de l'utilisateur ; `chosen`/`rejected` sont les sorties completes de l'assistant.
pub fn load_pairs(tok: &Tokenizer, text: &str) -> Result<Vec<Pair>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let v = json::parse(line).map_err(|e| format!("ligne {} : {}", n + 1, e))?;
        let get = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string());
        let (prompt, chosen, rejected) = (get("prompt").ok_or(format!("ligne {} : 'prompt' absent", n + 1))?, get("chosen").ok_or(format!("ligne {} : 'chosen' absent", n + 1))?, get("rejected").ok_or(format!("ligne {} : 'rejected' absent", n + 1))?);
        let mut ctx = String::new();
        if let Some(sys) = get("system") {
            ctx.push_str(&format!("<|system|>\n{}\n</|system|>\n", sys));
        }
        ctx.push_str(&format!("<|user|>\n{}\n</|user|>\n", prompt));
        let (c, r) = (tok.encode(chosen.trim_end()), tok.encode(rejected.trim_end()));
        if c.is_empty() || r.is_empty() || c == r {
            continue;
        }
        out.push(Pair { prompt: tok.encode(&ctx), chosen: c, rejected: r });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Airtel {
        let text = "bonjour le monde, merci beaucoup. hello world, thank you very much! ".repeat(20);
        let tok = Tokenizer::train_with(&text, 300, PreTok::Gpt);
        let cfg = AConfig { vocab: 0, d: 16, layers: 2, heads: 2, ff: 24, ctx: 40, rope_theta: 10000.0, eps: 1e-5 };
        Airtel::new(cfg, tok, 3).unwrap()
    }

    fn seq(m: &Airtel) -> (Vec<u32>, Vec<f32>) {
        let ids = m.tok.encode("<|user|>\nbonjour le monde\n</|user|>\n<|assistant|>\nhello world merci<|eos|>");
        let w: Vec<f32> = (0..ids.len()).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();
        (ids, w)
    }

    fn loss(m: &Airtel, ids: &[u32], w: &[f32]) -> f32 {
        let mut g = m.grads_full();
        m.seq_grad(&mut g, ids, w, &mut Rng::new(1)).0
    }

    #[test]
    fn gradient_complet() {
        let mut m = small();
        let (ids, w) = seq(&m);
        let mut g = m.grads_full();
        m.seq_grad(&mut g, &ids, &w, &mut Rng::new(1));
        let mut rng = Rng::new(9);
        let mut worst = 0.0f32;
        for ti in 0..m.p.len() {
            for _ in 0..4 {
                let k = rng.below(m.p[ti].data.len());
                let (orig, eps) = (m.p[ti].data[k], 2e-2f32);
                m.p[ti].data[k] = orig + eps;
                let lp = loss(&m, &ids, &w);
                m.p[ti].data[k] = orig - eps;
                let lm_ = loss(&m, &ids, &w);
                m.p[ti].data[k] = orig;
                let num = (lp - lm_) / (2.0 * eps);
                let ana = g.base[ti].data[k];
                let err = (num - ana).abs() / (num.abs() + ana.abs() + 0.05);
                worst = worst.max(err);
                assert!(err < 0.05, "tenseur {} idx {} : numerique {} analytique {}", ti, k, num, ana);
            }
        }
        eprintln!("pire erreur relative : {}", worst);
    }

    #[test]
    fn gradient_lora() {
        let mut m = small();
        m.add_lora(LoraCfg { r: 4, alpha: 8.0, dropout: 0.0, targets: LORA_KINDS.iter().map(|s| s.to_string()).collect() }, 5).unwrap();
        let mut rng = Rng::new(2);
        for t in m.lora.as_mut().unwrap().b.iter_mut() {
            t.data.iter_mut().for_each(|v| *v = rng.normal() * 0.1);
        }
        let (ids, w) = seq(&m);
        let f = |m: &Airtel| {
            let mut g = m.grads_lora();
            m.seq_grad(&mut g, &ids, &w, &mut Rng::new(1)).0
        };
        let mut g = m.grads_lora();
        m.seq_grad(&mut g, &ids, &w, &mut Rng::new(1));
        for s in (0..m.lora.as_ref().unwrap().a.len()).step_by(3) {
            for which in 0..2 {
                let n = if which == 0 { m.lora.as_ref().unwrap().a[s].data.len() } else { m.lora.as_ref().unwrap().b[s].data.len() };
                let k = rng.below(n);
                fn set(m: &mut Airtel, which: usize, s: usize, k: usize, v: Option<f32>) -> f32 {
                    let lo = m.lora.as_mut().unwrap();
                    let r = if which == 0 { &mut lo.a[s].data[k] } else { &mut lo.b[s].data[k] };
                    let old = *r;
                    if let Some(v) = v {
                        *r = v;
                    }
                    old
                }
                let orig = set(&mut m, which, s, k, None);
                let eps = 2e-2f32;
                set(&mut m, which, s, k, Some(orig + eps));
                let lp = f(&m);
                set(&mut m, which, s, k, Some(orig - eps));
                let lmn = f(&m);
                set(&mut m, which, s, k, Some(orig));
                let num = (lp - lmn) / (2.0 * eps);
                let ana = if which == 0 { g.la[s].data[k] } else { g.lb[s].data[k] };
                let err = (num - ana).abs() / (num.abs() + ana.abs() + 0.05);
                assert!(err < 0.05, "lora {} {} : num {} ana {}", s, which, num, ana);
            }
        }
    }

    #[test]
    fn cache_kv_egal_passe_complete() {
        let mut m = small();
        m.add_lora(LoraCfg { r: 2, alpha: 4.0, dropout: 0.0, targets: vec!["q_proj".into(), "down_proj".into()] }, 5).unwrap();
        let mut rng = Rng::new(4);
        for t in m.lora.as_mut().unwrap().b.iter_mut() {
            t.data.iter_mut().for_each(|v| *v = rng.normal() * 0.2);
        }
        let ids = m.tok.encode("<|user|>\nbonjour le monde\n</|user|>\n");
        let lg = m.last_logits(&ids);
        let fw = m.forward(&ids, false, &mut Rng::new(0));
        let (d, t) = (m.cfg.d, ids.len());
        let h = &fw.hf.data[(t - 1) * d..t * d];
        for v in 0..m.cfg.vocab {
            let full = dot(h, &m.p[EMB].data[v * d..(v + 1) * d]);
            assert!((full - lg[v]).abs() < 1e-3, "logit {} : {} vs {}", v, full, lg[v]);
        }
    }

    #[test]
    fn fusion_lora_equivalente() {
        let mut m = small();
        m.add_lora(LoraCfg { r: 4, alpha: 8.0, dropout: 0.0, targets: LORA_KINDS.iter().map(|s| s.to_string()).collect() }, 5).unwrap();
        let mut rng = Rng::new(4);
        for t in m.lora.as_mut().unwrap().b.iter_mut() {
            t.data.iter_mut().for_each(|v| *v = rng.normal() * 0.2);
        }
        let ids = m.tok.encode("bonjour le monde");
        let a = m.last_logits(&ids);
        m.merge_lora().unwrap();
        let b = m.last_logits(&ids);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-3);
        }
    }

    #[test]
    fn sauvegarde_aller_retour_et_hf() {
        let m = small();
        let dir = std::env::temp_dir().join(format!("airtel_t_{}", std::process::id()));
        let d = dir.to_str().unwrap();
        m.export_dir(d, "test").unwrap();
        let ids = m.tok.encode("bonjour le monde");
        let ref_l = m.last_logits(&ids);
        for src in [d.to_string(), format!("{}/model.safetensors", d)] {
            let m2 = Airtel::load(&src).unwrap();
            assert_eq!(m2.cfg, m.cfg);
            assert_eq!(m2.tok.encode("bonjour le monde"), ids);
            let l2 = m2.last_logits(&ids);
            assert!(ref_l.iter().zip(l2.iter()).all(|(a, b)| (a - b).abs() < 1e-6));
        }
        let (ts, _) = io::load_tensors(&format!("{}/model.safetensors", d)).unwrap();
        let q = ts.iter().find(|(n, _)| n == "model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q.1.shape, vec![16, 16]);
        let g = ts.iter().find(|(n, _)| n == "model.layers.0.mlp.gate_proj.weight").unwrap();
        assert_eq!(g.1.shape, vec![24, 16]); // [sortie, entree] comme nn.Linear
        assert!(ts.iter().all(|(n, _)| n != "lm_head.weight"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn apprentissage_et_dpo() {
        let mut m = small();
        let ids = m.tok.encode("<|user|>\nbonjour\n</|user|>\n<|assistant|>\nBonjour ! Merci.<|eos|>");
        let n = ids.len();
        let s = Sample { ids: ids.clone(), w: vec![1.0; n] };
        let before = m.eval_loss(&[s.clone()]);
        let o = TrainOpts { steps: 150, batch: 2, lr: 5e-3, warmup: 5, log_every: 0, eval_every: 0, threads: 1, ..TrainOpts::default() };
        m.train(Mode::Full, &[s.clone()], &[], &o, &mut |_| {}).unwrap();
        let after = m.eval_loss(&[s]);
        assert!(after < before * 0.4, "{} -> {}", before, after);
        let pair = Pair { prompt: m.tok.encode("<|user|>\nbonjour\n</|user|>\n"), chosen: m.tok.encode("<|assistant|>\nBonjour ! Merci.<|eos|>"), rejected: m.tok.encode("<|assistant|>\nhello world<|eos|>") };
        let dop = DpoOpts { epochs: 40.0, batch: 1, lr: 3e-3, threads: 1, log_every: 1000, ..DpoOpts::default() };
        let r = m.dpo(&[pair], &dop, &mut |_| {}).unwrap();
        assert!(r.accuracy == 1.0 && r.margin > 0.0, "{:?}", (r.accuracy, r.margin));
    }
}
