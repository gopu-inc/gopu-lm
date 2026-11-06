//! Tenseur f32 2D minimal + generateur pseudo-aleatoire.

/// Produit scalaire avec 8 accumulateurs independants (auto-vectorisable, bien plus rapide qu'une somme simple).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let ca = a[..n].chunks_exact(8);
    let cb = b[..n].chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0.0f32; 8];
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb.iter()) {
        s += x * y;
    }
    s
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor { data: vec![0.0; n], shape: shape.to_vec() }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Tensor, String> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(format!("forme {:?} incompatible avec {} valeurs", shape, data.len()));
        }
        Ok(Tensor { data, shape: shape.to_vec() })
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// self [m,k] x o [k,n] -> [m,n]
    pub fn matmul(&self, o: &Tensor) -> Tensor {
        let (m, k, n) = (self.rows(), self.cols(), o.cols());
        assert_eq!(k, o.rows(), "matmul: dimensions incompatibles");
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                if a == 0.0 {
                    continue;
                }
                let row = &o.data[p * n..(p + 1) * n];
                let dst = &mut out[i * n..(i + 1) * n];
                for j in 0..n {
                    dst[j] += a * row[j];
                }
            }
        }
        Tensor { data: out, shape: vec![m, n] }
    }

    /// self^T [k,m] . o [m,n] -> [k,n]   (self est [m,k])
    pub fn t_matmul(&self, o: &Tensor) -> Tensor {
        let (m, k, n) = (self.rows(), self.cols(), o.cols());
        assert_eq!(m, o.rows(), "t_matmul: dimensions incompatibles");
        let mut out = vec![0.0f32; k * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                if a == 0.0 {
                    continue;
                }
                for j in 0..n {
                    out[p * n + j] += a * o.data[i * n + j];
                }
            }
        }
        Tensor { data: out, shape: vec![k, n] }
    }

    /// self [m,n] . o^T [n,k] -> [m,k]   (o est [k,n])
    pub fn matmul_t(&self, o: &Tensor) -> Tensor {
        let (m, n, k) = (self.rows(), self.cols(), o.rows());
        assert_eq!(n, o.cols(), "matmul_t: dimensions incompatibles");
        let mut out = vec![0.0f32; m * k];
        for i in 0..m {
            for p in 0..k {
                out[i * k + p] = dot(&self.data[i * n..(i + 1) * n], &o.data[p * n..(p + 1) * n]);
            }
        }
        Tensor { data: out, shape: vec![m, k] }
    }
}

/// xorshift64* : suffisant pour initialiser des poids et melanger des indices.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1) ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n.max(1) as u64)) as usize
    }
}
