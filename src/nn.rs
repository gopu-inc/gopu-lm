//! Reseau dense (MLP) avec retropropagation manuelle, AdamW et entrainement par mini-batch.
//! Convention des poids : w est [entree, sortie] (different de PyTorch qui stocke [sortie, entree]).

use crate::tensor::{Rng, Tensor};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    None,
    Relu,
    Tanh,
}

impl Act {
    pub fn name(&self) -> &'static str {
        match self {
            Act::None => "none",
            Act::Relu => "relu",
            Act::Tanh => "tanh",
        }
    }
    pub fn parse(s: &str) -> Result<Act, String> {
        match s {
            "none" | "linear" | "" => Ok(Act::None),
            "relu" => Ok(Act::Relu),
            "tanh" => Ok(Act::Tanh),
            other => Err(format!("activation inconnue : {}", other)),
        }
    }
}

pub struct Dense {
    pub w: Tensor,
    pub b: Tensor,
    pub act: Act,
    gw: Tensor,
    gb: Tensor,
    mw: Tensor,
    vw: Tensor,
    mb: Tensor,
    vb: Tensor,
    x: Tensor,
    a: Tensor,
}

impl Dense {
    fn new(i: usize, o: usize, act: Act, rng: &mut Rng) -> Dense {
        let std = (2.0f32 / (i + o) as f32).sqrt();
        let mut w = Tensor::zeros(&[i, o]);
        for v in w.data.iter_mut() {
            *v = rng.normal() * std;
        }
        Dense {
            w,
            b: Tensor::zeros(&[o]),
            act,
            gw: Tensor::zeros(&[i, o]),
            gb: Tensor::zeros(&[o]),
            mw: Tensor::zeros(&[i, o]),
            vw: Tensor::zeros(&[i, o]),
            mb: Tensor::zeros(&[o]),
            vb: Tensor::zeros(&[o]),
            x: Tensor::zeros(&[0, i]),
            a: Tensor::zeros(&[0, o]),
        }
    }

    fn forward(&mut self, x: &Tensor) -> Tensor {
        let n = self.w.cols();
        let mut y = x.matmul(&self.w);
        for r in 0..y.rows() {
            for j in 0..n {
                y.data[r * n + j] += self.b.data[j];
            }
        }
        match self.act {
            Act::None => {}
            Act::Relu => {
                for v in y.data.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Act::Tanh => {
                for v in y.data.iter_mut() {
                    *v = v.tanh();
                }
            }
        }
        self.x = x.clone();
        self.a = y.clone();
        y
    }

    fn backward(&mut self, dout: &Tensor) -> Tensor {
        let mut dp = dout.clone();
        match self.act {
            Act::None => {}
            Act::Relu => {
                for (d, a) in dp.data.iter_mut().zip(self.a.data.iter()) {
                    if *a <= 0.0 {
                        *d = 0.0;
                    }
                }
            }
            Act::Tanh => {
                for (d, a) in dp.data.iter_mut().zip(self.a.data.iter()) {
                    *d *= 1.0 - a * a;
                }
            }
        }
        let gw = self.x.t_matmul(&dp);
        for (g, v) in self.gw.data.iter_mut().zip(gw.data.iter()) {
            *g += *v;
        }
        let n = self.w.cols();
        for r in 0..dp.rows() {
            for j in 0..n {
                self.gb.data[j] += dp.data[r * n + j];
            }
        }
        dp.matmul_t(&self.w)
    }
}

fn adam(p: &mut [f32], g: &[f32], m: &mut [f32], v: &mut [f32], lr: f32, wd: f32, scale: f32, t: i32) {
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let c1 = 1.0 - b1.powi(t);
    let c2 = 1.0 - b2.powi(t);
    for i in 0..p.len() {
        let gi = g[i] * scale;
        m[i] = b1 * m[i] + (1.0 - b1) * gi;
        v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
        let mh = m[i] / c1;
        let vh = v[i] / c2;
        p[i] -= lr * (mh / (vh.sqrt() + eps) + wd * p[i]);
    }
}

/// Softmax + entropie croisee. Retourne (perte moyenne, gradient sur les logits).
pub fn softmax_ce(logits: &Tensor, labels: &[usize]) -> Result<(f32, Tensor), String> {
    let (m, n) = (logits.rows(), logits.cols());
    if labels.len() != m {
        return Err(format!("{} etiquettes pour {} lignes", labels.len(), m));
    }
    let mut grad = Tensor::zeros(&[m, n]);
    let mut loss = 0.0f32;
    for i in 0..m {
        if labels[i] >= n {
            return Err(format!("etiquette {} hors limites (classes : {})", labels[i], n));
        }
        let row = &logits.data[i * n..(i + 1) * n];
        let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..n {
            let e = (row[j] - mx).exp();
            grad.data[i * n + j] = e;
            sum += e;
        }
        for j in 0..n {
            grad.data[i * n + j] /= sum;
        }
        loss += -(grad.data[i * n + labels[i]] + 1e-12).ln();
        grad.data[i * n + labels[i]] -= 1.0;
    }
    for v in grad.data.iter_mut() {
        *v /= m as f32;
    }
    Ok((loss / m as f32, grad))
}

pub struct Mlp {
    pub layers: Vec<Dense>,
    pub dims: Vec<usize>,
    t: i32,
    rng: Rng,
}

impl Mlp {
    pub fn new(dims: &[usize], acts: &[Act], seed: u64) -> Result<Mlp, String> {
        if dims.len() < 2 || acts.len() != dims.len() - 1 {
            return Err("il faut au moins 2 dimensions et une activation par couche".to_string());
        }
        if dims.iter().any(|&d| d == 0) {
            return Err("dimension nulle".to_string());
        }
        let mut rng = Rng::new(seed);
        let mut layers = Vec::new();
        for i in 0..dims.len() - 1 {
            layers.push(Dense::new(dims[i], dims[i + 1], acts[i], &mut rng));
        }
        Ok(Mlp { layers, dims: dims.to_vec(), t: 0, rng })
    }

    pub fn in_dim(&self) -> usize {
        self.dims[0]
    }
    pub fn out_dim(&self) -> usize {
        self.dims[self.dims.len() - 1]
    }

    pub fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut h = x.clone();
        for l in self.layers.iter_mut() {
            h = l.forward(&h);
        }
        h
    }

    fn step(&mut self, lr: f32, wd: f32, clip: f32) {
        self.t += 1;
        let mut sq = 0.0f32;
        for l in self.layers.iter() {
            sq += l.gw.data.iter().map(|g| g * g).sum::<f32>();
            sq += l.gb.data.iter().map(|g| g * g).sum::<f32>();
        }
        let norm = sq.sqrt();
        let scale = if clip > 0.0 && norm > clip { clip / norm } else { 1.0 };
        let t = self.t;
        for l in self.layers.iter_mut() {
            adam(&mut l.w.data, &l.gw.data, &mut l.mw.data, &mut l.vw.data, lr, wd, scale, t);
            adam(&mut l.b.data, &l.gb.data, &mut l.mb.data, &mut l.vb.data, lr, 0.0, scale, t);
            for g in l.gw.data.iter_mut() {
                *g = 0.0;
            }
            for g in l.gb.data.iter_mut() {
                *g = 0.0;
            }
        }
    }

    pub fn train_step_ce(&mut self, x: &Tensor, labels: &[usize], lr: f32) -> Result<f32, String> {
        let logits = self.forward(x);
        let (loss, grad) = softmax_ce(&logits, labels)?;
        let mut g = grad;
        for l in self.layers.iter_mut().rev() {
            g = l.backward(&g);
        }
        self.step(lr, 0.01, 1.0);
        Ok(loss)
    }

    /// Entrainement par mini-batch avec melange a chaque epoque. Retourne la perte moyenne par epoque.
    pub fn fit(&mut self, x: &Tensor, labels: &[usize], epochs: usize, lr: f32, batch: usize) -> Result<Vec<f32>, String> {
        let n = x.rows();
        if x.cols() != self.in_dim() {
            return Err(format!("entree de taille {} attendue, recu {}", self.in_dim(), x.cols()));
        }
        if labels.len() != n || n == 0 {
            return Err("etiquettes/donnees incoherentes".to_string());
        }
        let batch = batch.max(1).min(n);
        let d = x.cols();
        let mut idx: Vec<usize> = (0..n).collect();
        let mut history = Vec::new();
        for _ in 0..epochs {
            for i in (1..n).rev() {
                let j = self.rng.below(i + 1);
                idx.swap(i, j);
            }
            let mut total = 0.0f32;
            let mut count = 0usize;
            for chunk in idx.chunks(batch) {
                let mut data = Vec::with_capacity(chunk.len() * d);
                let mut lab = Vec::with_capacity(chunk.len());
                for &i in chunk {
                    data.extend_from_slice(&x.data[i * d..(i + 1) * d]);
                    lab.push(labels[i]);
                }
                let xb = Tensor::from_vec(data, &[chunk.len(), d])?;
                total += self.train_step_ce(&xb, &lab, lr)?;
                count += 1;
            }
            history.push(total / count as f32);
        }
        Ok(history)
    }

    pub fn export(&self) -> (Vec<(String, Tensor)>, Vec<(String, String)>) {
        let mut t = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            t.push((format!("layers.{}.weight", i), l.w.clone()));
            t.push((format!("layers.{}.bias", i), l.b.clone()));
        }
        let dims: Vec<String> = self.dims.iter().map(|d| d.to_string()).collect();
        let acts: Vec<&str> = self.layers.iter().map(|l| l.act.name()).collect();
        let meta = vec![
            ("arch".to_string(), "gopu-mlp".to_string()),
            ("dims".to_string(), dims.join(",")),
            ("acts".to_string(), acts.join(",")),
            ("framework".to_string(), "gopu 0.1.0".to_string()),
        ];
        (t, meta)
    }

    pub fn import(tensors: &[(String, Tensor)], meta: &[(String, String)]) -> Result<Mlp, String> {
        let find = |k: &str| meta.iter().find(|(mk, _)| mk == k).map(|(_, v)| v.clone());
        let arch = find("arch").ok_or("metadonnee 'arch' absente")?;
        if arch != "gopu-mlp" {
            return Err(format!("architecture non supportee : {}", arch));
        }
        let dims_s = find("dims").ok_or("metadonnee 'dims' absente")?;
        let acts_s = find("acts").ok_or("metadonnee 'acts' absente")?;
        let dims: Vec<usize> = dims_s
            .split(',')
            .map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<Vec<usize>, String>>()?;
        let acts: Vec<Act> = acts_s
            .split(',')
            .map(|s| Act::parse(s.trim()))
            .collect::<Result<Vec<Act>, String>>()?;
        let mut m = Mlp::new(&dims, &acts, 1)?;
        for (i, l) in m.layers.iter_mut().enumerate() {
            let wn = format!("layers.{}.weight", i);
            let bn = format!("layers.{}.bias", i);
            let w = tensors.iter().find(|(k, _)| *k == wn).ok_or(format!("tenseur {} absent", wn))?;
            let b = tensors.iter().find(|(k, _)| *k == bn).ok_or(format!("tenseur {} absent", bn))?;
            if w.1.shape != l.w.shape || b.1.shape != l.b.shape {
                return Err(format!("forme inattendue pour la couche {}", i));
            }
            l.w = w.1.clone();
            l.b = b.1.clone();
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_numerique() {
        let mut m = Mlp::new(&[3, 4, 2], &[Act::Tanh, Act::None], 7).unwrap();
        let x = Tensor::from_vec(vec![0.1, -0.2, 0.3, 0.5, 0.4, -0.6], &[2, 3]).unwrap();
        let labels = [0usize, 1usize];
        let logits = m.forward(&x);
        let (_, g) = softmax_ce(&logits, &labels).unwrap();
        let mut gg = g;
        for l in m.layers.iter_mut().rev() {
            gg = l.backward(&gg);
        }
        let analytic = m.layers[0].gw.data[2];
        let eps = 1e-3f32;
        let loss_at = |m: &mut Mlp, delta: f32| -> f32 {
            m.layers[0].w.data[2] += delta;
            let lg = m.forward(&x);
            let (l, _) = softmax_ce(&lg, &labels).unwrap();
            m.layers[0].w.data[2] -= delta;
            l
        };
        let num = (loss_at(&mut m, eps) - loss_at(&mut m, -eps)) / (2.0 * eps);
        assert!(
            (analytic - num).abs() < 1e-2 * (1.0 + num.abs()),
            "analytique {} vs numerique {}",
            analytic,
            num
        );
    }
}
