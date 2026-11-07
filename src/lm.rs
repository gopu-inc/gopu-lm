//! Modele de langage gopu-lm : Transformer GPT "pre-norm" en Rust pur (std uniquement),
//! retropropagation ecrite a la main.
//!
//! Architecture
//!  - embeddings de tokens [V,d] + embeddings de position appris [ctx,d]
//!  - `layers` blocs : LayerNorm -> attention causale multi-tetes -> residuel
//!                     LayerNorm -> MLP (GELU) -> residuel
//!  - LayerNorm finale, puis tete de sortie dont les poids sont PARTAGES avec les embeddings
//!  - `future` tetes de "prediction du futur" (multi-token prediction) : la tete j predit le token t+2+j
//!    a partir de l'etat cache du token t (petit bloc residuel + sortie partagee). Elles servent
//!    d'objectif auxiliaire a l'entrainement et exposent `predict_future` a l'inference.

use crate::io;
use crate::tensor::{dot, Rng, Tensor};
use crate::tokenizer::{self, Tokenizer};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub vocab: usize,
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ctx: usize,
    pub ff: usize,
    pub future: usize,
}

impl Config {
    pub fn small(vocab: usize) -> Config {
        Config { vocab, d: 64, layers: 3, heads: 4, ctx: 128, ff: 256, future: 2 }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.d == 0 || self.layers == 0 || self.heads == 0 || self.ctx < 2 || self.ff == 0 || self.vocab < 2 {
            return Err("configuration invalide : dimensions nulles".to_string());
        }
        if self.d % self.heads != 0 {
            return Err(format!("d ({}) doit etre divisible par heads ({})", self.d, self.heads));
        }
        Ok(())
    }

    fn n_tensors(&self) -> usize {
        4 + 12 * self.layers + 2 * self.future
    }

    pub fn names(&self) -> Vec<String> {
        let mut n = vec!["tok_emb".to_string(), "pos_emb".to_string()];
        for l in 0..self.layers {
            for k in ["ln1.weight", "ln1.bias", "attn.wq", "attn.wk", "attn.wv", "attn.wo", "ln2.weight", "ln2.bias", "mlp.w1", "mlp.b1", "mlp.w2", "mlp.b2"] {
                n.push(format!("layers.{}.{}", l, k));
            }
        }
        n.push("ln_f.weight".to_string());
        n.push("ln_f.bias".to_string());
        for j in 0..self.future {
            n.push(format!("future.{}.weight", j));
            n.push(format!("future.{}.bias", j));
        }
        n
    }

    pub fn shapes(&self) -> Vec<Vec<usize>> {
        let (v, d, f) = (self.vocab, self.d, self.ff);
        let mut s = vec![vec![v, d], vec![self.ctx, d]];
        for _ in 0..self.layers {
            s.extend([vec![d], vec![d], vec![d, d], vec![d, d], vec![d, d], vec![d, d], vec![d], vec![d], vec![d, f], vec![f], vec![f, d], vec![d]]);
        }
        s.push(vec![d]);
        s.push(vec![d]);
        for _ in 0..self.future {
            s.push(vec![d, d]);
            s.push(vec![d]);
        }
        s
    }

    pub fn n_params(&self) -> usize {
        self.shapes().iter().map(|s| s.iter().product::<usize>()).sum()
    }
}

const TOK: usize = 0;
const POS: usize = 1;
// ordre dans un bloc : 0 ln1.g 1 ln1.b 2 wq 3 wk 4 wv 5 wo 6 ln2.g 7 ln2.b 8 w1 9 b1 10 w2 11 b2
fn lay(l: usize, k: usize) -> usize {
    2 + l * 12 + k
}
fn lnf(c: &Config) -> usize {
    2 + 12 * c.layers
}
fn fut(c: &Config, j: usize) -> usize {
    4 + 12 * c.layers + 2 * j
}

pub struct Params {
    pub cfg: Config,
    pub t: Vec<Tensor>,
}

impl Params {
    pub fn init(cfg: &Config, seed: u64) -> Params {
        let mut rng = Rng::new(seed);
        let names = cfg.names();
        let shapes = cfg.shapes();
        let resid = 1.0 / (2.0 * cfg.layers as f32).sqrt();
        let mut t = Vec::new();
        for (name, shape) in names.iter().zip(shapes.iter()) {
            let mut x = Tensor::zeros(shape);
            if name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name == "ln_f.weight" {
                x.data.iter_mut().for_each(|v| *v = 1.0);
            } else if shape.len() == 2 {
                let std = if name.ends_with("attn.wo") || name.ends_with("mlp.w2") { 0.02 * resid } else { 0.02 };
                x.data.iter_mut().for_each(|v| *v = rng.normal() * std);
            }
            t.push(x);
        }
        Params { cfg: cfg.clone(), t }
    }

    fn zeros_like(&self) -> Vec<Tensor> {
        self.t.iter().map(|x| Tensor::zeros(&x.shape)).collect()
    }
}

// ------------------------------------------------------------------ operations de base

fn add_assign(dst: &mut Tensor, src: &Tensor) {
    for (a, b) in dst.data.iter_mut().zip(src.data.iter()) {
        *a += *b;
    }
}

fn add_bias(y: &mut Tensor, b: &Tensor) {
    let n = y.cols();
    for r in 0..y.rows() {
        for j in 0..n {
            y.data[r * n + j] += b.data[j];
        }
    }
}

fn colsum_add(dst: &mut [f32], dy: &Tensor) {
    let n = dy.cols();
    for r in 0..dy.rows() {
        for j in 0..n {
            dst[j] += dy.data[r * n + j];
        }
    }
}

const LN_EPS: f32 = 1e-5;

fn layer_norm(x: &Tensor, g: &Tensor, b: &Tensor) -> (Tensor, Tensor, Vec<f32>) {
    let (t, d) = (x.rows(), x.cols());
    let mut y = Tensor::zeros(&[t, d]);
    let mut xhat = Tensor::zeros(&[t, d]);
    let mut rstd = vec![0.0f32; t];
    for r in 0..t {
        let row = &x.data[r * d..(r + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
        let rs = 1.0 / (var + LN_EPS).sqrt();
        rstd[r] = rs;
        for j in 0..d {
            let h = (row[j] - mean) * rs;
            xhat.data[r * d + j] = h;
            y.data[r * d + j] = h * g.data[j] + b.data[j];
        }
    }
    (y, xhat, rstd)
}

/// Retourne (dx, dgain, dbias).
fn layer_norm_bwd(dy: &Tensor, xhat: &Tensor, rstd: &[f32], g: &Tensor) -> (Tensor, Vec<f32>, Vec<f32>) {
    let (t, d) = (dy.rows(), dy.cols());
    let mut dx = Tensor::zeros(&[t, d]);
    let mut dg = vec![0.0f32; d];
    let mut db = vec![0.0f32; d];
    for r in 0..t {
        let (mut m1, mut m2) = (0.0f32, 0.0f32);
        for j in 0..d {
            let dxh = dy.data[r * d + j] * g.data[j];
            m1 += dxh;
            m2 += dxh * xhat.data[r * d + j];
            dg[j] += dy.data[r * d + j] * xhat.data[r * d + j];
            db[j] += dy.data[r * d + j];
        }
        m1 /= d as f32;
        m2 /= d as f32;
        for j in 0..d {
            let dxh = dy.data[r * d + j] * g.data[j];
            dx.data[r * d + j] = rstd[r] * (dxh - m1 - xhat.data[r * d + j] * m2);
        }
    }
    (dx, dg, db)
}

const GELU_C: f32 = 0.797_884_560_8; // sqrt(2/pi)

#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_C * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    let u = GELU_C * (x + 0.044715 * x * x * x);
    let t = u.tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * 0.044715 * x * x)
}

// ------------------------------------------------------------------ attention causale multi-tetes

fn attn_forward(q: &Tensor, k: &Tensor, v: &Tensor, heads: usize) -> (Tensor, Vec<f32>) {
    let (t, d) = (q.rows(), q.cols());
    let dh = d / heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut p = vec![0.0f32; heads * t * t];
    let mut out = Tensor::zeros(&[t, d]);
    for h in 0..heads {
        let off = h * dh;
        for i in 0..t {
            let row = &mut p[(h * t + i) * t..(h * t + i) * t + t];
            let qi = &q.data[i * d + off..i * d + off + dh];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..=i {
                let s = dot(qi, &k.data[j * d + off..j * d + off + dh]) * scale;
                row[j] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..=i {
                row[j] = (row[j] - mx).exp();
                sum += row[j];
            }
            let inv = 1.0 / sum;
            let oi = &mut out.data[i * d + off..i * d + off + dh];
            for j in 0..=i {
                row[j] *= inv;
                let w = row[j];
                let vj = &v.data[j * d + off..j * d + off + dh];
                for c in 0..dh {
                    oi[c] += w * vj[c];
                }
            }
        }
    }
    (out, p)
}

fn attn_backward(datt: &Tensor, q: &Tensor, k: &Tensor, v: &Tensor, p: &[f32], heads: usize) -> (Tensor, Tensor, Tensor) {
    let (t, d) = (q.rows(), q.cols());
    let dh = d / heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut dq = Tensor::zeros(&[t, d]);
    let mut dk = Tensor::zeros(&[t, d]);
    let mut dv = Tensor::zeros(&[t, d]);
    let mut dp = vec![0.0f32; t];
    for h in 0..heads {
        let off = h * dh;
        for i in 0..t {
            let prow = &p[(h * t + i) * t..(h * t + i) * t + t];
            let doi = &datt.data[i * d + off..i * d + off + dh];
            let mut s = 0.0f32;
            for j in 0..=i {
                dp[j] = dot(doi, &v.data[j * d + off..j * d + off + dh]);
                s += prow[j] * dp[j];
                let dvj = &mut dv.data[j * d + off..j * d + off + dh];
                for c in 0..dh {
                    dvj[c] += prow[j] * doi[c];
                }
            }
            for j in 0..=i {
                let ds = prow[j] * (dp[j] - s) * scale;
                let kj = &k.data[j * d + off..j * d + off + dh];
                let qi = &q.data[i * d + off..i * d + off + dh];
                for c in 0..dh {
                    dq.data[i * d + off + c] += ds * kj[c];
                    dk.data[j * d + off + c] += ds * qi[c];
                }
            }
        }
    }
    (dq, dk, dv)
}

// ------------------------------------------------------------------ passe avant

struct Cache {
    y1: Tensor,
    xhat1: Tensor,
    rstd1: Vec<f32>,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    p: Vec<f32>,
    att: Tensor,
    y2: Tensor,
    xhat2: Tensor,
    rstd2: Vec<f32>,
    hp: Tensor,
    ha: Tensor,
}

struct Fwd {
    caches: Vec<Cache>,
    xhatf: Tensor,
    rstdf: Vec<f32>,
    yf: Tensor,
}

fn forward(p: &Params, ids: &[u32], keep: bool) -> Fwd {
    let c = &p.cfg;
    let (t, d) = (ids.len(), c.d);
    let mut x = Tensor::zeros(&[t, d]);
    for (i, &id) in ids.iter().enumerate() {
        let te = &p.t[TOK].data[id as usize * d..(id as usize + 1) * d];
        let pe = &p.t[POS].data[i * d..(i + 1) * d];
        for j in 0..d {
            x.data[i * d + j] = te[j] + pe[j];
        }
    }
    let mut caches = Vec::with_capacity(c.layers);
    for l in 0..c.layers {
        let (y1, xhat1, rstd1) = layer_norm(&x, &p.t[lay(l, 0)], &p.t[lay(l, 1)]);
        let q = y1.matmul(&p.t[lay(l, 2)]);
        let k = y1.matmul(&p.t[lay(l, 3)]);
        let v = y1.matmul(&p.t[lay(l, 4)]);
        let (att, pr) = attn_forward(&q, &k, &v, c.heads);
        let o = att.matmul(&p.t[lay(l, 5)]);
        let mut x2 = x;
        add_assign(&mut x2, &o);
        let (y2, xhat2, rstd2) = layer_norm(&x2, &p.t[lay(l, 6)], &p.t[lay(l, 7)]);
        let mut hp = y2.matmul(&p.t[lay(l, 8)]);
        add_bias(&mut hp, &p.t[lay(l, 9)]);
        let mut ha = hp.clone();
        ha.data.iter_mut().for_each(|v| *v = gelu(*v));
        let mut m = ha.matmul(&p.t[lay(l, 10)]);
        add_bias(&mut m, &p.t[lay(l, 11)]);
        x = x2;
        add_assign(&mut x, &m);
        caches.push(Cache { y1, xhat1, rstd1, q, k, v, p: if keep { pr } else { Vec::new() }, att, y2, xhat2, rstd2, hp, ha });
    }
    let (yf, xhatf, rstdf) = layer_norm(&x, &p.t[lnf(c)], &p.t[lnf(c) + 1]);
    Fwd { caches, xhatf, rstdf, yf }
}

/// Bloc de la tete j de prediction du futur : retourne (z, u) avec u = h + gelu(h.W + b).
fn future_block(p: &Params, h: &Tensor, j: usize) -> (Tensor, Tensor) {
    let i = fut(&p.cfg, j);
    let mut z = h.matmul(&p.t[i]);
    add_bias(&mut z, &p.t[i + 1]);
    let mut u = h.clone();
    for (uu, zz) in u.data.iter_mut().zip(z.data.iter()) {
        *uu += gelu(*zz);
    }
    (z, u)
}

/// Softmax + entropie croisee ponderee, sur place. Chaque ligne devient le gradient sur les logits.
/// Retourne (perte ponderee * coeff, somme des poids).
fn ce_rows(logits: &mut Tensor, targets: &[Option<(usize, f32)>], coeff: f32) -> (f32, f32) {
    let n = logits.cols();
    let (mut loss, mut ws) = (0.0f32, 0.0f32);
    for (r, tg) in targets.iter().enumerate() {
        let row = &mut logits.data[r * n..(r + 1) * n];
        match tg {
            Some((t, wt)) if *wt > 0.0 => {
                let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in row.iter_mut() {
                    *x = (*x - mx).exp();
                    sum += *x;
                }
                let inv = 1.0 / sum;
                for x in row.iter_mut() {
                    *x *= inv;
                }
                loss += coeff * wt * -(row[*t].max(1e-12)).ln();
                ws += wt;
                let sc = coeff * wt;
                for x in row.iter_mut() {
                    *x *= sc;
                }
                row[*t] -= sc;
            }
            _ => row.iter_mut().for_each(|x| *x = 0.0),
        }
    }
    (loss, ws)
}

/// Perte principale seule (evaluation). Retourne (perte ponderee, somme des poids).
fn main_loss(p: &Params, seq: &[u32], w: &[f32]) -> (f32, f32) {
    let n = seq.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let t = n - 1;
    let fw = forward(p, &seq[..t], false);
    let mut logits = fw.yf.matmul_t(&p.t[TOK]);
    let targets: Vec<Option<(usize, f32)>> = (0..t).map(|r| Some((seq[r + 1] as usize, w[r + 1]))).collect();
    ce_rows(&mut logits, &targets, 1.0)
}

/// Perte + gradients (accumules dans `g`) pour UNE sequence de n jetons.
/// `w[i]` est le poids du jeton i quand il est la cible. Retourne (perte totale, somme des poids principaux) ;
/// les gradients ne sont PAS normalises (le diviseur est choisi par l'appelant).
fn seq_grad(p: &Params, g: &mut [Tensor], seq: &[u32], w: &[f32], lambda: f32) -> (f32, f32) {
    let c = &p.cfg;
    let n = seq.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let t = n - 1;
    let d = c.d;
    let fw = forward(p, &seq[..t], true);
    let tok = &p.t[TOK];
    let (mut loss, mut wsum) = (0.0f32, 0.0f32);
    let mut dyf = Tensor::zeros(&[t, d]);

    // tete principale (prochain jeton)
    {
        let mut dl = fw.yf.matmul_t(tok);
        let targets: Vec<Option<(usize, f32)>> = (0..t).map(|r| Some((seq[r + 1] as usize, w[r + 1]))).collect();
        let (l, ws) = ce_rows(&mut dl, &targets, 1.0);
        loss += l;
        wsum += ws;
        add_assign(&mut dyf, &dl.matmul(tok));
        add_assign(&mut g[TOK], &dl.t_matmul(&fw.yf));
    }
    // tetes de prediction du futur : token t+2+j
    for j in 0..c.future {
        let s = j + 2;
        if s >= n {
            break;
        }
        let (z, u) = future_block(p, &fw.yf, j);
        let mut dl = u.matmul_t(tok);
        let targets: Vec<Option<(usize, f32)>> = (0..t).map(|r| if r + s < n { Some((seq[r + s] as usize, w[r + s])) } else { None }).collect();
        let (l, _) = ce_rows(&mut dl, &targets, lambda);
        loss += l;
        let du = dl.matmul(tok);
        add_assign(&mut g[TOK], &dl.t_matmul(&u));
        add_assign(&mut dyf, &du);
        let mut dz = du;
        for (dd, zz) in dz.data.iter_mut().zip(z.data.iter()) {
            *dd *= gelu_grad(*zz);
        }
        let i = fut(c, j);
        add_assign(&mut g[i], &fw.yf.t_matmul(&dz));
        colsum_add(&mut g[i + 1].data, &dz);
        add_assign(&mut dyf, &dz.matmul_t(&p.t[i]));
    }

    // LayerNorm finale
    let (mut dx, dg, db) = layer_norm_bwd(&dyf, &fw.xhatf, &fw.rstdf, &p.t[lnf(c)]);
    g[lnf(c)].data.iter_mut().zip(dg.iter()).for_each(|(a, b)| *a += *b);
    g[lnf(c) + 1].data.iter_mut().zip(db.iter()).for_each(|(a, b)| *a += *b);

    for l in (0..c.layers).rev() {
        let ch = &fw.caches[l];
        // MLP : x_out = x2 + ha.w2 + b2
        add_assign(&mut g[lay(l, 10)], &ch.ha.t_matmul(&dx));
        colsum_add(&mut g[lay(l, 11)].data, &dx);
        let mut dhp = dx.matmul_t(&p.t[lay(l, 10)]);
        for (dd, hh) in dhp.data.iter_mut().zip(ch.hp.data.iter()) {
            *dd *= gelu_grad(*hh);
        }
        add_assign(&mut g[lay(l, 8)], &ch.y2.t_matmul(&dhp));
        colsum_add(&mut g[lay(l, 9)].data, &dhp);
        let dy2 = dhp.matmul_t(&p.t[lay(l, 8)]);
        let (dxn2, dg2, db2) = layer_norm_bwd(&dy2, &ch.xhat2, &ch.rstd2, &p.t[lay(l, 6)]);
        g[lay(l, 6)].data.iter_mut().zip(dg2.iter()).for_each(|(a, b)| *a += *b);
        g[lay(l, 7)].data.iter_mut().zip(db2.iter()).for_each(|(a, b)| *a += *b);
        let mut dx2 = dx;
        add_assign(&mut dx2, &dxn2);
        // attention : x2 = x + att.wo
        add_assign(&mut g[lay(l, 5)], &ch.att.t_matmul(&dx2));
        let datt = dx2.matmul_t(&p.t[lay(l, 5)]);
        let (dq, dk, dv) = attn_backward(&datt, &ch.q, &ch.k, &ch.v, &ch.p, c.heads);
        add_assign(&mut g[lay(l, 2)], &ch.y1.t_matmul(&dq));
        add_assign(&mut g[lay(l, 3)], &ch.y1.t_matmul(&dk));
        add_assign(&mut g[lay(l, 4)], &ch.y1.t_matmul(&dv));
        let mut dy1 = dq.matmul_t(&p.t[lay(l, 2)]);
        add_assign(&mut dy1, &dk.matmul_t(&p.t[lay(l, 3)]));
        add_assign(&mut dy1, &dv.matmul_t(&p.t[lay(l, 4)]));
        let (dxn1, dg1, db1) = layer_norm_bwd(&dy1, &ch.xhat1, &ch.rstd1, &p.t[lay(l, 0)]);
        g[lay(l, 0)].data.iter_mut().zip(dg1.iter()).for_each(|(a, b)| *a += *b);
        g[lay(l, 1)].data.iter_mut().zip(db1.iter()).for_each(|(a, b)| *a += *b);
        dx = dx2;
        add_assign(&mut dx, &dxn1);
    }

    // embeddings
    for (i, &id) in seq[..t].iter().enumerate() {
        let id = id as usize;
        for j in 0..d {
            let v = dx.data[i * d + j];
            g[TOK].data[id * d + j] += v;
            g[POS].data[i * d + j] += v;
        }
    }
    (loss, wsum)
}

// ------------------------------------------------------------------ donnees d'entrainement

pub struct Sample {
    pub ids: Vec<u32>,
    pub w: Vec<f32>,
}

/// Transforme un corpus de conversations (separees par `<|eos|>`) en exemples avec poids de perte :
///  - tour utilisateur / systeme (balises comprises) : `user_weight` (0 = on ne l'apprend pas)
///  - resultats d'outils (injectes par le programme, pas par le modele) : poids 0
///  - tout le reste (reflexion, reponse, appel d'outil, fin de tour) : poids 1
pub fn build_samples(tok: &Tokenizer, text: &str, user_weight: f32) -> Vec<Sample> {
    let ids = tok.encode(text);
    let mut out = Vec::new();
    let mut cur: Vec<u32> = Vec::new();
    for &id in &ids {
        cur.push(id);
        if id == tokenizer::EOS {
            push_sample(&mut out, std::mem::take(&mut cur), user_weight);
        }
    }
    push_sample(&mut out, cur, user_weight);
    out
}

fn push_sample(out: &mut Vec<Sample>, ids: Vec<u32>, uw: f32) {
    // retire les retours a la ligne de tete (separateurs entre conversations)
    let start = ids.iter().position(|&i| i != tokenizer::NEWLINE).unwrap_or(ids.len());
    let ids: Vec<u32> = ids[start..].to_vec();
    if ids.len() < 4 {
        return;
    }
    #[derive(PartialEq)]
    enum R {
        Out,
        Prompt,
        Result,
    }
    let mut region = R::Out;
    let mut skip_nl = false;
    let mut w = Vec::with_capacity(ids.len());
    for &id in &ids {
        let wt = match id {
            tokenizer::USER_OPEN | tokenizer::SYSTEM_OPEN => {
                region = R::Prompt;
                uw
            }
            tokenizer::USER_CLOSE | tokenizer::SYSTEM_CLOSE => {
                region = R::Out;
                uw
            }
            tokenizer::RESULT_OPEN => {
                region = R::Result;
                skip_nl = false;
                0.0
            }
            tokenizer::RESULT_CLOSE => {
                region = R::Out;
                skip_nl = true;
                0.0
            }
            tokenizer::TOOL_CLOSE => {
                skip_nl = true;
                1.0
            }
            _ => match region {
                R::Prompt => uw,
                R::Result => 0.0,
                R::Out => {
                    if skip_nl && id == tokenizer::NEWLINE {
                        skip_nl = false;
                        0.0
                    } else {
                        skip_nl = false;
                        1.0
                    }
                }
            },
        };
        w.push(wt);
    }
    out.push(Sample { ids, w });
}

/// Preparation du corpus.
#[derive(Clone, Debug)]
pub struct DataOpts {
    pub user_weight: f32,
    /// 1 exemple sur N part en validation (0 ou 1 = pas de validation)
    pub holdout_every: usize,
    /// plafond de jetons d'entrainement (0 = illimite)
    pub max_corpus_tokens: usize,
    /// longueur cible des sequences : plusieurs conversations sont mises bout a bout (0 = une conversation par sequence)
    pub pack_len: usize,
    pub seed: u64,
}

impl Default for DataOpts {
    fn default() -> Self {
        DataOpts { user_weight: 0.0, holdout_every: 20, max_corpus_tokens: 0, pack_len: 0, seed: 1 }
    }
}

pub struct DataStats {
    pub conversations: usize,
    pub train_tokens: usize,
    pub train_sequences: usize,
    pub val_sequences: usize,
    pub truncated: bool,
}

/// texte -> (exemples d'entrainement, exemples de validation) : decoupe, validation, melange, plafond de jetons, empaquetage.
pub fn prepare_data(tok: &Tokenizer, text: &str, o: &DataOpts) -> (Vec<Sample>, Vec<Sample>, DataStats) {
    let all = build_samples(tok, text, o.user_weight);
    let conversations = all.len();
    let (mut tr, mut va) = (Vec::new(), Vec::new());
    for (i, s) in all.into_iter().enumerate() {
        if o.holdout_every > 1 && i % o.holdout_every == o.holdout_every - 1 {
            va.push(s);
        } else {
            tr.push(s);
        }
    }
    let mut rng = Rng::new(o.seed);
    for i in (1..tr.len()).rev() {
        let j = rng.below(i + 1);
        tr.swap(i, j);
    }
    let mut truncated = false;
    if o.max_corpus_tokens > 0 {
        let mut total = 0usize;
        let mut keep = 0usize;
        for s in &tr {
            if total >= o.max_corpus_tokens {
                truncated = true;
                break;
            }
            total += s.ids.len();
            keep += 1;
        }
        tr.truncate(keep);
    }
    if o.pack_len > 0 {
        let mut packed: Vec<Sample> = Vec::new();
        let mut cur = Sample { ids: Vec::new(), w: Vec::new() };
        for s in tr {
            if !cur.ids.is_empty() && cur.ids.len() + s.ids.len() > o.pack_len + 1 {
                packed.push(std::mem::replace(&mut cur, Sample { ids: Vec::new(), w: Vec::new() }));
            }
            cur.ids.extend(s.ids);
            cur.w.extend(s.w);
        }
        if !cur.ids.is_empty() {
            packed.push(cur);
        }
        tr = packed;
    }
    let stats = DataStats { conversations, train_tokens: tr.iter().map(|s| s.ids.len()).sum(), train_sequences: tr.len(), val_sequences: va.len(), truncated };
    (tr, va, stats)
}

// ------------------------------------------------------------------ entrainement

#[derive(Clone, Debug)]
pub struct TrainOpts {
    pub steps: usize,
    pub batch: usize,
    pub lr: f32,
    pub min_lr_frac: f32,
    pub warmup: usize,
    pub wd: f32,
    pub clip: f32,
    pub future_weight: f32,
    pub seed: u64,
    pub log_every: usize,
    pub eval_every: usize,
    pub threads: usize,
    pub max_seconds: f32,
}

impl Default for TrainOpts {
    fn default() -> Self {
        TrainOpts {
            steps: 1000,
            batch: 16,
            lr: 3e-3,
            min_lr_frac: 0.1,
            warmup: 40,
            wd: 0.01,
            clip: 1.0,
            future_weight: 0.3,
            seed: 1,
            log_every: 50,
            eval_every: 200,
            threads: 0,
            max_seconds: 0.0,
        }
    }
}

pub struct TrainReport {
    pub steps: usize,
    pub loss: f32,
    pub val_loss: Option<f32>,
    pub seconds: f32,
}

/// Adaptateur LoRA de la tete de sortie partagee. Le modele de base reste gele ;
/// l'adaptateur ajoute `alpha / rank * h.A.B` aux logits.
#[derive(Clone)]
pub struct Lora {
    pub rank: usize,
    pub alpha: f32,
    pub a: Tensor, // [d, rank]
    pub b: Tensor, // [rank, vocab]
}

impl Lora {
    fn new(cfg: &Config, rank: usize, alpha: f32, seed: u64) -> Result<Self, String> {
        if rank == 0 {
            return Err("le rang LoRA doit etre superieur a zero".to_string());
        }
        let mut rng = Rng::new(seed);
        let mut a = Tensor::zeros(&[cfg.d, rank]);
        for x in &mut a.data {
            *x = rng.normal() * 0.02;
        }
        Ok(Lora { rank, alpha: if alpha > 0.0 { alpha } else { rank as f32 }, a, b: Tensor::zeros(&[rank, cfg.vocab]) })
    }

    fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    fn add_logits(&self, h: &[f32], logits: &mut [f32]) {
        let mut z = vec![0.0f32; self.rank];
        for i in 0..h.len() {
            for k in 0..self.rank {
                z[k] += h[i] * self.a.data[i * self.rank + k];
            }
        }
        let scale = self.scale();
        let vocab = logits.len();
        for (v, logit) in logits.iter_mut().enumerate() {
            for k in 0..self.rank {
                *logit += scale * z[k] * self.b.data[k * vocab + v];
            }
        }
    }

    fn save(&self, cfg: &Config, path: &str) -> Result<(), String> {
        let tensors = vec![("output_lora.A".to_string(), self.a.clone()), ("output_lora.B".to_string(), self.b.clone())];
        let meta = vec![
            ("arch".to_string(), "gopu-lora".to_string()),
            ("target".to_string(), "output".to_string()),
            ("d".to_string(), cfg.d.to_string()),
            ("vocab".to_string(), cfg.vocab.to_string()),
            ("rank".to_string(), self.rank.to_string()),
            ("alpha".to_string(), self.alpha.to_string()),
        ];
        io::save_tensors(path, &tensors, &meta)
    }

    fn load(cfg: &Config, path: &str) -> Result<Self, String> {
        let (tensors, meta) = io::load_tensors(path)?;
        let find = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.clone());
        if find("arch").as_deref() != Some("gopu-lora") || find("target").as_deref() != Some("output") {
            return Err("ce fichier n'est pas un adaptateur LoRA de sortie gopu".to_string());
        }
        let number = |k: &str| -> Result<usize, String> { find(k).ok_or(format!("metadonnee LoRA '{}' absente", k))?.parse().map_err(|_| format!("metadonnee LoRA '{}' invalide", k)) };
        if number("d")? != cfg.d || number("vocab")? != cfg.vocab {
            return Err("cet adaptateur LoRA ne correspond pas a l'architecture du modele".to_string());
        }
        let rank = number("rank")?;
        let alpha = find("alpha").ok_or("metadonnee LoRA 'alpha' absente")?.parse::<f32>().map_err(|_| "metadonnee LoRA 'alpha' invalide")?;
        let tensor = |name: &str| -> Result<Tensor, String> { tensors.iter().find(|(n, _)| n == name).map(|(_, x)| x.clone()).ok_or(format!("tenseur LoRA '{}' absent", name)) };
        let a = tensor("output_lora.A")?;
        let b = tensor("output_lora.B")?;
        if a.shape != vec![cfg.d, rank] || b.shape != vec![rank, cfg.vocab] {
            return Err("formes LoRA incompatibles avec le modele".to_string());
        }
        Ok(Lora { rank, alpha, a, b })
    }
}

fn crop<'a>(s: &'a Sample, ctx: usize, rng: &mut Rng) -> (&'a [u32], &'a [f32]) {
    let n = s.ids.len();
    if n <= ctx + 1 {
        (&s.ids[..], &s.w[..])
    } else {
        let st = rng.below(n - ctx);
        (&s.ids[st..st + ctx + 1], &s.w[st..st + ctx + 1])
    }
}

// ------------------------------------------------------------------ inference

#[derive(Clone, Debug)]
pub struct GenOpts {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub stop_ids: Vec<u32>,
    pub stop_strings: Vec<String>,
}

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts {
            max_tokens: 160,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.0,
            seed: 0,
            stop_ids: vec![tokenizer::ASSIST_CLOSE, tokenizer::EOS, tokenizer::TOOL_CLOSE],
            stop_strings: Vec::new(),
        }
    }
}

pub struct GenOut {
    pub tokens: Vec<u32>,
    /// Pourquoi la generation s'est arretee : "stop_token", "stop_string" ou "length".
    pub finish: &'static str,
    /// Texte du jeton d'arret, s'il y en a un.
    pub stop: Option<String>,
}

pub struct Lm {
    pub params: Params,
    pub tok: Tokenizer,
    pub lora: Option<Lora>,
}

impl Lm {
    pub fn new(mut cfg: Config, tok: Tokenizer, seed: u64) -> Result<Lm, String> {
        cfg.vocab = tok.vocab_size();
        cfg.validate()?;
        Ok(Lm { params: Params::init(&cfg, seed), tok, lora: None })
    }

    pub fn cfg(&self) -> &Config {
        &self.params.cfg
    }

    // ---- sauvegarde / chargement (.b8n ou .safetensors, selon l'extension)

    pub fn save(&self, path: &str) -> Result<(), String> {
        let c = self.cfg();
        let tensors: Vec<(String, Tensor)> = c.names().into_iter().zip(self.params.t.iter().cloned()).collect();
        let meta = vec![
            ("arch".to_string(), "gopu-lm".to_string()),
            ("vocab".to_string(), c.vocab.to_string()),
            ("d".to_string(), c.d.to_string()),
            ("layers".to_string(), c.layers.to_string()),
            ("heads".to_string(), c.heads.to_string()),
            ("ctx".to_string(), c.ctx.to_string()),
            ("ff".to_string(), c.ff.to_string()),
            ("future".to_string(), c.future.to_string()),
            ("n_specials".to_string(), tokenizer::SPECIALS.len().to_string()),
            ("tok_merges".to_string(), self.tok.merges_to_string()),
            ("framework".to_string(), "gopu 0.2.0".to_string()),
        ];
        io::save_tensors(path, &tensors, &meta)
    }

    pub fn load(path: &str) -> Result<Lm, String> {
        let (tensors, meta) = io::load_tensors(path)?;
        let find = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.clone());
        match find("arch").as_deref() {
            Some("gopu-lm") => {}
            Some(other) => return Err(format!("ce fichier contient une architecture '{}', pas un gopu-lm", other)),
            None => return Err("metadonnee 'arch' absente".to_string()),
        }
        let num = |k: &str| -> Result<usize, String> {
            find(k).ok_or(format!("metadonnee '{}' absente", k))?.parse::<usize>().map_err(|e| format!("'{}' : {}", k, e))
        };
        let cfg = Config { vocab: num("vocab")?, d: num("d")?, layers: num("layers")?, heads: num("heads")?, ctx: num("ctx")?, ff: num("ff")?, future: num("future")? };
        cfg.validate()?;
        if num("n_specials")? != tokenizer::SPECIALS.len() {
            return Err("jeu de jetons speciaux incompatible avec cette version".to_string());
        }
        let tok = Tokenizer::merges_from_string(&find("tok_merges").unwrap_or_default())?;
        if tok.vocab_size() != cfg.vocab {
            return Err(format!("vocabulaire incoherent : tokenizer {} vs modele {}", tok.vocab_size(), cfg.vocab));
        }
        let mut t = Vec::new();
        for (name, shape) in cfg.names().iter().zip(cfg.shapes().iter()) {
            let (_, x) = tensors.iter().find(|(k, _)| k == name).ok_or(format!("tenseur '{}' absent", name))?;
            if &x.shape != shape {
                return Err(format!("forme inattendue pour '{}' : {:?} au lieu de {:?}", name, x.shape, shape));
            }
            t.push(x.clone());
        }
        debug_assert_eq!(t.len(), cfg.n_tensors());
        Ok(Lm { params: Params { cfg, t }, tok, lora: None })
    }

    /// Change la taille de contexte. Les positions existantes sont conservees ; les nouvelles sont initialisees
    /// au hasard (petites valeurs) et devront etre apprises par l'entrainement.
    pub fn resize_ctx(&mut self, new_ctx: usize) -> Result<(), String> {
        if new_ctx < 2 {
            return Err("contexte trop petit".to_string());
        }
        let d = self.cfg().d;
        let old = self.params.t[POS].clone();
        let mut rng = Rng::new(99);
        let mut n = Tensor::zeros(&[new_ctx, d]);
        for i in 0..new_ctx * d {
            n.data[i] = if i < old.data.len() { old.data[i] } else { rng.normal() * 0.02 };
        }
        self.params.t[POS] = n;
        self.params.cfg.ctx = new_ctx;
        Ok(())
    }

    /// Active un nouvel adaptateur LoRA de rang `rank` sur la tete de sortie.
    pub fn enable_lora(&mut self, rank: usize, alpha: f32, seed: u64) -> Result<(), String> {
        self.lora = Some(Lora::new(self.cfg(), rank, alpha, seed)?);
        Ok(())
    }

    pub fn save_lora(&self, path: &str) -> Result<(), String> {
        self.lora.as_ref().ok_or("aucun adaptateur LoRA actif")?.save(self.cfg(), path)
    }

    pub fn load_lora(&mut self, path: &str) -> Result<(), String> {
        self.lora = Some(Lora::load(self.cfg(), path)?);
        Ok(())
    }

    // ---- entrainement

    pub fn train(&mut self, train: &[Sample], val: &[Sample], o: &TrainOpts, log: &mut dyn FnMut(&str)) -> Result<TrainReport, String> {
        if train.is_empty() {
            return Err("aucun exemple d'entrainement".to_string());
        }
        let ctx = self.cfg().ctx;
        let vocab = self.cfg().vocab as u32;
        if train.iter().any(|s| s.ids.iter().any(|&i| i >= vocab)) {
            return Err("un jeton depasse le vocabulaire du modele".to_string());
        }
        let nthreads = if o.threads > 0 { o.threads } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) };
        let nthreads = nthreads.min(o.batch.max(1));
        let mut rng = Rng::new(o.seed);
        let mut m = self.params.zeros_like();
        let mut v = self.params.zeros_like();
        let start = Instant::now();
        let mut ema: Option<f32> = None;
        let mut last_val = None;
        let mut done = 0usize;
        for step in 1..=o.steps {
            // taux d'apprentissage : montee lineaire puis cosinus jusqu'a min_lr_frac * lr
            let lr = if step <= o.warmup {
                o.lr * step as f32 / o.warmup.max(1) as f32
            } else {
                let prog = (step - o.warmup) as f32 / (o.steps - o.warmup).max(1) as f32;
                let cos = 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
                o.lr * (o.min_lr_frac + (1.0 - o.min_lr_frac) * cos)
            };
            // tirage du mini-lot (avec recadrage aleatoire des exemples trop longs)
            let mut batch: Vec<(&[u32], &[f32])> = Vec::with_capacity(o.batch);
            for _ in 0..o.batch {
                let s = &train[rng.below(train.len())];
                batch.push(crop(s, ctx, &mut rng));
            }
            let params = &self.params;
            let chunk = (batch.len() + nthreads - 1) / nthreads;
            let results: Vec<(Vec<Tensor>, f32, f32)> = std::thread::scope(|sc| {
                let handles: Vec<_> = batch
                    .chunks(chunk)
                    .map(|part| {
                        sc.spawn(move || {
                            let mut g = params.zeros_like();
                            let (mut l, mut w) = (0.0f32, 0.0f32);
                            for (ids, ws) in part {
                                let (a, b) = seq_grad(params, &mut g, ids, ws, o.future_weight);
                                l += a;
                                w += b;
                            }
                            (g, l, w)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().expect("thread d'entrainement")).collect()
            });
            let mut it = results.into_iter();
            let (mut g, mut loss, mut wsum) = it.next().ok_or("lot vide")?;
            for (gi, li, wi) in it {
                for (a, b) in g.iter_mut().zip(gi.iter()) {
                    add_assign(a, b);
                }
                loss += li;
                wsum += wi;
            }
            if wsum <= 0.0 {
                continue;
            }
            let inv = 1.0 / wsum;
            let mut sq = 0.0f32;
            for gt in &g {
                sq += gt.data.iter().map(|x| (x * inv) * (x * inv)).sum::<f32>();
            }
            let norm = sq.sqrt();
            if !norm.is_finite() {
                return Err(format!("gradient non fini a l'etape {} : entrainement instable, baisse lr", step));
            }
            let scale = inv * if o.clip > 0.0 && norm > o.clip { o.clip / norm } else { 1.0 };
            adamw(&mut self.params, &g, &mut m, &mut v, lr, o.wd, scale, step as i32);
            done = step;
            let main = loss / wsum;
            ema = Some(match ema {
                Some(e) => 0.95 * e + 0.05 * main,
                None => main,
            });
            if o.eval_every > 0 && !val.is_empty() && (step % o.eval_every == 0 || step == o.steps) {
                last_val = Some(self.eval_loss(val));
            }
            if o.log_every > 0 && (step % o.log_every == 0 || step == 1 || step == o.steps) {
                let vs = last_val.map(|x| format!(" | val {:.3}", x)).unwrap_or_default();
                log(&format!("etape {:>5}/{} | perte {:.3} (lissee {:.3}){} | lr {:.5} | |g| {:.2} | {:.0}s", step, o.steps, main, ema.unwrap_or(main), vs, lr, norm, start.elapsed().as_secs_f32()));
            }
            if o.max_seconds > 0.0 && start.elapsed().as_secs_f32() > o.max_seconds {
                log(&format!("temps maximum atteint ({:.0}s) : arret a l'etape {}", o.max_seconds, step));
                break;
            }
        }
        if !val.is_empty() {
            last_val = Some(self.eval_loss(val));
        }
        Ok(TrainReport { steps: done, loss: ema.unwrap_or(f32::NAN), val_loss: last_val, seconds: start.elapsed().as_secs_f32() })
    }

    /// Entraine seulement les matrices LoRA de la tete de sortie. Les parametres du Transformer ne sont jamais modifies.
    pub fn train_lora(&mut self, train: &[Sample], val: &[Sample], o: &TrainOpts, rank: usize, alpha: f32, log: &mut dyn FnMut(&str)) -> Result<TrainReport, String> {
        if train.is_empty() {
            return Err("aucun exemple d'entrainement LoRA".to_string());
        }
        if self.lora.is_none() {
            self.enable_lora(rank, alpha, o.seed)?;
        }
        let cfg = self.cfg().clone();
        let lora = self.lora.as_mut().unwrap();
        if lora.rank != rank {
            return Err(format!("rang LoRA actif {} different du rang demande {}", lora.rank, rank));
        }
        let (mut ma, mut va) = (Tensor::zeros(&lora.a.shape), Tensor::zeros(&lora.a.shape));
        let (mut mb, mut vb) = (Tensor::zeros(&lora.b.shape), Tensor::zeros(&lora.b.shape));
        let mut rng = Rng::new(o.seed);
        let start = Instant::now();
        let mut ema = None;
        let mut done = 0;
        for step in 1..=o.steps {
            let (mut ga, mut gb) = (Tensor::zeros(&lora.a.shape), Tensor::zeros(&lora.b.shape));
            let (mut total_loss, mut total_weight) = (0.0f32, 0.0f32);
            for _ in 0..o.batch.max(1) {
                let s = &train[rng.below(train.len())];
                let (seq, weights) = crop(s, cfg.ctx, &mut rng);
                if seq.len() < 2 { continue; }
                let t = seq.len() - 1;
                let fw = forward(&self.params, &seq[..t], false);
                for r in 0..t {
                    let wt = weights[r + 1];
                    if wt <= 0.0 { continue; }
                    let h = &fw.yf.data[r * cfg.d..(r + 1) * cfg.d];
                    let mut z = vec![0.0f32; lora.rank];
                    for i in 0..cfg.d {
                        for k in 0..lora.rank { z[k] += h[i] * lora.a.data[i * lora.rank + k]; }
                    }
                    let mut logits: Vec<f32> = (0..cfg.vocab).map(|v| dot(h, &self.params.t[TOK].data[v * cfg.d..(v + 1) * cfg.d])).collect();
                    let scale = lora.scale();
                    for v in 0..cfg.vocab {
                        for k in 0..lora.rank { logits[v] += scale * z[k] * lora.b.data[k * cfg.vocab + v]; }
                    }
                    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for x in &mut logits { *x = (*x - mx).exp(); sum += *x; }
                    let target = seq[r + 1] as usize;
                    total_loss += wt * -(logits[target] / sum).max(1e-12).ln();
                    total_weight += wt;
                    let mut gz = vec![0.0f32; lora.rank];
                    for v in 0..cfg.vocab {
                        let dl = wt * (logits[v] / sum - if v == target { 1.0 } else { 0.0 });
                        for k in 0..lora.rank {
                            gb.data[k * cfg.vocab + v] += scale * z[k] * dl;
                            gz[k] += scale * lora.b.data[k * cfg.vocab + v] * dl;
                        }
                    }
                    for i in 0..cfg.d {
                        for k in 0..lora.rank { ga.data[i * lora.rank + k] += h[i] * gz[k]; }
                    }
                }
            }
            if total_weight == 0.0 { continue; }
            let lr = if step <= o.warmup { o.lr * step as f32 / o.warmup.max(1) as f32 } else { o.lr };
            let inv = 1.0 / total_weight;
            adam_tensor(&mut lora.a, &ga, &mut ma, &mut va, lr, o.wd, inv, step as i32);
            adam_tensor(&mut lora.b, &gb, &mut mb, &mut vb, lr, 0.0, inv, step as i32);
            let loss = total_loss * inv;
            ema = Some(match ema { Some(x) => 0.95 * x + 0.05 * loss, None => loss });
            done = step;
            if o.log_every > 0 && (step == 1 || step % o.log_every == 0 || step == o.steps) {
                log(&format!("LoRA etape {:>5}/{} | perte {:.3} (lissee {:.3}) | {:.0}s", step, o.steps, loss, ema.unwrap_or(loss), start.elapsed().as_secs_f32()));
            }
            if o.max_seconds > 0.0 && start.elapsed().as_secs_f32() > o.max_seconds { break; }
        }
        let val_loss = if val.is_empty() { None } else { Some(self.eval_lora_loss(val)) };
        Ok(TrainReport { steps: done, loss: ema.unwrap_or(f32::NAN), val_loss, seconds: start.elapsed().as_secs_f32() })
    }

    fn eval_lora_loss(&self, samples: &[Sample]) -> f32 {
        let lora = match &self.lora { Some(x) => x, None => return f32::NAN };
        let cfg = self.cfg();
        let mut rng = Rng::new(7);
        let (mut loss, mut weight) = (0.0f32, 0.0f32);
        for s in samples {
            let (seq, weights) = crop(s, cfg.ctx, &mut rng);
            if seq.len() < 2 { continue; }
            let fw = forward(&self.params, &seq[..seq.len() - 1], false);
            for r in 0..seq.len() - 1 {
                let wt = weights[r + 1];
                if wt <= 0.0 { continue; }
                let h = &fw.yf.data[r * cfg.d..(r + 1) * cfg.d];
                let mut logits: Vec<f32> = (0..cfg.vocab).map(|v| dot(h, &self.params.t[TOK].data[v * cfg.d..(v + 1) * cfg.d])).collect();
                lora.add_logits(h, &mut logits);
                let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = logits.iter().map(|x| (*x - mx).exp()).sum();
                loss += wt * -((logits[seq[r + 1] as usize] - mx).exp() / sum).max(1e-12).ln();
                weight += wt;
            }
        }
        loss / weight.max(1.0)
    }

    /// Perte moyenne (jeton suivant, ponderee) sur des exemples.
    pub fn eval_loss(&self, samples: &[Sample]) -> f32 {
        let ctx = self.cfg().ctx;
        let mut rng = Rng::new(7);
        let (mut l, mut w) = (0.0f32, 0.0f32);
        for s in samples {
            let (ids, ws) = crop(s, ctx, &mut rng);
            let (a, b) = main_loss(&self.params, ids, ws);
            l += a;
            w += b;
        }
        if w > 0.0 {
            l / w
        } else {
            f32::NAN
        }
    }

    // ---- inference

    fn window<'a>(&self, ids: &'a [u32]) -> &'a [u32] {
        let ctx = self.cfg().ctx;
        if ids.len() > ctx {
            &ids[ids.len() - ctx..]
        } else {
            ids
        }
    }

    /// Logits du prochain jeton (position la plus recente de la fenetre de contexte).
    pub fn last_logits(&self, ids: &[u32]) -> Vec<f32> {
        let ids = self.window(ids);
        // L'API publique peut recevoir directement des ids arbitraires via le FFI.
        // Ne laissons pas une entrée vide ou hors vocabulaire provoquer un panic.
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= self.cfg().vocab) {
            return vec![0.0; self.cfg().vocab];
        }
        let fw = forward(&self.params, ids, false);
        let d = self.cfg().d;
        let last = &fw.yf.data[(ids.len() - 1) * d..ids.len() * d];
        let tok = &self.params.t[TOK];
        let mut logits: Vec<f32> = (0..self.cfg().vocab).map(|v| dot(last, &tok.data[v * d..(v + 1) * d])).collect();
        if let Some(lora) = &self.lora {
            lora.add_logits(last, &mut logits);
        }
        logits
    }

    /// Distribution des jetons a venir : element 0 = jeton suivant, element j+1 = jeton t+2+j (tete j).
    /// Pour chaque horizon, les `top_n` jetons les plus probables avec leur probabilite.
    pub fn predict_future(&self, ids: &[u32], top_n: usize) -> Vec<Vec<(u32, f32)>> {
        let ids = self.window(ids);
        if ids.is_empty() {
            return Vec::new();
        }
        let fw = forward(&self.params, ids, false);
        let (d, c) = (self.cfg().d, self.cfg());
        let tok = &self.params.t[TOK];
        let last = Tensor { data: fw.yf.data[(ids.len() - 1) * d..ids.len() * d].to_vec(), shape: vec![1, d] };
        let mut outs = Vec::new();
        let mut heads: Vec<Tensor> = vec![last.clone()];
        for j in 0..c.future {
            heads.push(future_block(&self.params, &last, j).1);
        }
        for h in heads {
            let mut logits: Vec<f32> = (0..c.vocab).map(|v| dot(&h.data, &tok.data[v * d..(v + 1) * d])).collect();
            if let Some(lora) = &self.lora {
                lora.add_logits(&h.data, &mut logits);
            }
            logits[tokenizer::PAD as usize] = f32::NEG_INFINITY;
            outs.push(top_probs(&logits, top_n));
        }
        outs
    }

    /// Embeddings statiques des jetons (lignes de la matrice d'embedding), forme [n, d].
    pub fn token_embeddings(&self, ids: &[u32]) -> Tensor {
        let d = self.cfg().d;
        let mut t = Tensor::zeros(&[ids.len(), d]);
        for (i, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(self.cfg().vocab - 1);
            t.data[i * d..(i + 1) * d].copy_from_slice(&self.params.t[TOK].data[id * d..(id + 1) * d]);
        }
        t
    }

    /// Etats caches contextuels de la derniere couche, forme [n, d] (n limite a la fenetre de contexte).
    pub fn hidden_states(&self, ids: &[u32]) -> Tensor {
        let ids = self.window(ids);
        if ids.is_empty() {
            return Tensor::zeros(&[0, self.cfg().d]);
        }
        forward(&self.params, ids, false).yf
    }

    pub fn generate(&self, prompt: &[u32], o: &GenOpts) -> GenOut {
        let mut ids: Vec<u32> = prompt.to_vec();
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= self.cfg().vocab) {
            return GenOut { tokens: Vec::new(), finish: "invalid_prompt", stop: None };
        }
        let mut seed = o.seed;
        if seed == 0 {
            seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
        }
        let mut rng = Rng::new(seed);
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..o.max_tokens {
            let mut logits = self.last_logits(&ids);
            let recent_from = ids.len().saturating_sub(64);
            let next = sample(&mut logits, o, &mut rng, &ids[recent_from..]);
            ids.push(next);
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
        }
        GenOut { tokens: out, finish: "length", stop: None }
    }
}

fn top_probs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| (l - mx).exp()).sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter().take(n.max(1)).map(|i| (i as u32, (logits[i] - mx).exp() / sum)).collect()
}

fn sample(logits: &mut [f32], o: &GenOpts, rng: &mut Rng, recent: &[u32]) -> u32 {
    if logits.is_empty() {
        return tokenizer::PAD;
    }
    logits[tokenizer::PAD as usize] = f32::NEG_INFINITY;
    let repetition_penalty = if o.repetition_penalty.is_finite() && o.repetition_penalty > 0.0 { o.repetition_penalty } else { 1.0 };
    if (repetition_penalty - 1.0).abs() > 1e-6 {
        let mut seen: Vec<u32> = recent.to_vec();
        seen.sort_unstable();
        seen.dedup();
        for id in seen {
            let l = &mut logits[id as usize];
            *l = if *l > 0.0 { *l / repetition_penalty } else { *l * repetition_penalty };
        }
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    if !o.temperature.is_finite() || o.temperature <= 1e-6 {
        return idx[0] as u32;
    }
    let k = if o.top_k == 0 { idx.len() } else { o.top_k.min(idx.len()) };
    idx.truncate(k);
    let mx = logits[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| ((logits[i] - mx) / o.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    if o.top_p > 0.0 && o.top_p < 1.0 {
        let mut acc = 0.0f32;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            acc += p;
            if acc >= o.top_p {
                cut = i + 1;
                break;
            }
        }
        probs.truncate(cut);
        idx.truncate(cut);
        let s: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= s);
    }
    let r = rng.uniform();
    let mut acc = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return idx[i] as u32;
        }
    }
    idx[idx.len() - 1] as u32
}

/// AdamW (decouple) sur tous les tenseurs ; le decroissement de poids ne s'applique qu'aux matrices de couches.
fn adamw(p: &mut Params, g: &[Tensor], m: &mut [Tensor], v: &mut [Tensor], lr: f32, wd: f32, scale: f32, t: i32) {
    let (b1, b2, eps) = (0.9f32, 0.95f32, 1e-8f32);
    let c1 = 1.0 - b1.powi(t);
    let c2 = 1.0 - b2.powi(t);
    for i in 0..p.t.len() {
        let decay = if p.t[i].shape.len() == 2 && i != TOK && i != POS { wd } else { 0.0 };
        let (pi, gi, mi, vi) = (&mut p.t[i].data, &g[i].data, &mut m[i].data, &mut v[i].data);
        for k in 0..pi.len() {
            let gk = gi[k] * scale;
            mi[k] = b1 * mi[k] + (1.0 - b1) * gk;
            vi[k] = b2 * vi[k] + (1.0 - b2) * gk * gk;
            let step = (mi[k] / c1) / ((vi[k] / c2).sqrt() + eps);
            pi[k] -= lr * (step + decay * pi[k]);
        }
    }
}

/// AdamW applique a une unique matrice LoRA.
fn adam_tensor(p: &mut Tensor, g: &Tensor, m: &mut Tensor, v: &mut Tensor, lr: f32, wd: f32, scale: f32, t: i32) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let (c1, c2) = (1.0 - b1.powi(t), 1.0 - b2.powi(t));
    for i in 0..p.data.len() {
        let gi = g.data[i] * scale;
        m.data[i] = b1 * m.data[i] + (1.0 - b1) * gi;
        v.data[i] = b2 * v.data[i] + (1.0 - b2) * gi * gi;
        let step = (m.data[i] / c1) / ((v.data[i] / c2).sqrt() + eps);
        p.data[i] -= lr * (step + wd * p.data[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Lm {
        let tok = Tokenizer::train(&"<|user|>\nBonjour\n</|user|>\n<|assistant|>\nSalut !\n</|assistant|>\n<|eos|>\n".repeat(10), 280);
        let cfg = Config { vocab: 0, d: 8, layers: 2, heads: 2, ctx: 16, ff: 16, future: 2 };
        Lm::new(cfg, tok, 3).unwrap()
    }

    /// Gradient analytique (toute la chaine : embeddings, attention, LN, MLP, tetes du futur) contre differences finies.
    #[test]
    fn gradient_numerique() {
        let lm = tiny();
        let seq: Vec<u32> = vec![260, 270, 100, 257, 261, 105, 106, 270, 99, 10];
        let w: Vec<f32> = vec![0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let lambda = 0.3;
        let mut g = lm.params.zeros_like();
        let (_, ws) = seq_grad(&lm.params, &mut g, &seq, &w, lambda);
        assert!(ws > 0.0);
        let mut p = Params { cfg: lm.params.cfg.clone(), t: lm.params.t.clone() };
        let loss = |p: &Params| -> f32 { seq_grad(p, &mut p.zeros_like(), &seq, &w, lambda).0 };
        let mut rng = Rng::new(11);
        let mut worst = 0.0f32;
        for ti in 0..p.t.len() {
            for _ in 0..3 {
                let k = rng.below(p.t[ti].data.len());
                let eps = 1e-2f32;
                let orig = p.t[ti].data[k];
                p.t[ti].data[k] = orig + eps;
                let lp = loss(&p);
                p.t[ti].data[k] = orig - eps;
                let lm_ = loss(&p);
                p.t[ti].data[k] = orig;
                let num = (lp - lm_) / (2.0 * eps);
                let ana = g[ti].data[k];
                let err = (num - ana).abs() / (1e-2 + num.abs().max(ana.abs()));
                worst = worst.max(err);
                assert!(err < 0.1, "tenseur {} ({}) idx {} : analytique {} vs numerique {}", ti, lm.cfg().names()[ti], k, ana, num);
            }
        }
        println!("erreur relative max : {}", worst);
    }

    #[test]
    fn apprend_une_phrase() {
        let mut lm = tiny();
        let text = "<|user|>\nBonjour\n</|user|>\n<|assistant|>\nSalut !\n</|assistant|>\n<|eos|>\n";
        let samples = build_samples(&lm.tok, &text.repeat(4), 0.0);
        let mut o = TrainOpts::default();
        o.steps = 400;
        o.batch = 4;
        o.lr = 1e-2;
        o.warmup = 5;
        o.log_every = 0;
        o.eval_every = 0;
        lm.train(&samples, &[], &o, &mut |_| {}).unwrap();
        let fin = lm.eval_loss(&samples);
        assert!(fin < 0.1, "perte finale {}", fin);
        let prompt = lm.tok.encode("<|user|>\nBonjour\n</|user|>\n");
        let mut g = GenOpts::default();
        g.temperature = 0.0;
        let out = lm.generate(&prompt, &g);
        let txt = lm.tok.decode(&out.tokens);
        assert!(txt.contains("<|assistant|>\nSalut !"), "sortie : {:?}", txt);
    }

    #[test]
    fn sauvegarde_rechargement() {
        let lm = tiny();
        let path = std::env::temp_dir().join("gopu_lm_test.b8n");
        let path = path.to_str().unwrap();
        lm.save(path).unwrap();
        let lm2 = Lm::load(path).unwrap();
        let ids = lm.tok.encode("<|user|>\nBonjour\n</|user|>\n");
        assert_eq!(lm.last_logits(&ids), lm2.last_logits(&ids));
        assert_eq!(lm.cfg(), lm2.cfg());
    }

    #[test]
    fn poids_de_perte() {
        let tok = Tokenizer::new();
        let txt = "<|user|>\nA\n</|user|>\n<|assistant|>\n<|tools=\"python\">\nx\n</|tools>\n<|tool_result>\n7\n</|tool_result>\nok\n</|assistant|>\n<|eos|>\n";
        let s = build_samples(&tok, txt, 0.0);
        assert_eq!(s.len(), 1);
        let ids = &s[0].ids;
        let w = &s[0].w;
        for (i, &id) in ids.iter().enumerate() {
            if id == tokenizer::USER_OPEN || id == tokenizer::USER_CLOSE || id == tokenizer::RESULT_OPEN || id == tokenizer::RESULT_CLOSE {
                assert_eq!(w[i], 0.0, "balise {} doit etre ignoree", id);
            }
            if id == tokenizer::ASSIST_CLOSE || id == tokenizer::TOOL_CLOSE || id == tokenizer::EOS {
                assert_eq!(w[i], 1.0);
            }
        }
        assert!(w.iter().sum::<f32>() > 5.0);
    }
}
