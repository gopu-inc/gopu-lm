//! Tokenizer BPE octet par octet (entrainable) + jetons speciaux pour le format de chat gopu.
//!
//! - ids 0..255          : les 256 octets
//! - ids 256..256+S      : jetons speciaux (SPECIALS)
//! - ids suivants        : fusions BPE apprises (une fusion = deux ids -> un nouvel id)
//!
//! Pre-decoupage : un mot (lettres) garde son espace initial, chaque chiffre est un jeton a part
//! (utile pour recopier des nombres), toute autre ponctuation est un jeton a part.

use std::collections::HashMap;
use std::sync::Mutex;

/// Jetons speciaux du format de chat. L'ordre fixe leurs ids (256 + position).
pub const SPECIALS: &[&str] = &[
    "<|pad|>",
    "<|eos|>",
    "<|system|>",
    "</|system|>",
    "<|user|>",
    "</|user|>",
    "<|thinking|>",
    "</|thinking|>",
    "<|assistant|>",
    "</|assistant|>",
    "<|tools=\"",
    "</|tools>",
    "<|tool_result>",
    "</|tool_result>",
];

pub const BYTES: u32 = 256;
pub const PAD: u32 = BYTES;
pub const EOS: u32 = BYTES + 1;
pub const SYSTEM_OPEN: u32 = BYTES + 2;
pub const SYSTEM_CLOSE: u32 = BYTES + 3;
pub const USER_OPEN: u32 = BYTES + 4;
pub const USER_CLOSE: u32 = BYTES + 5;
pub const THINK_OPEN: u32 = BYTES + 6;
pub const THINK_CLOSE: u32 = BYTES + 7;
pub const ASSIST_OPEN: u32 = BYTES + 8;
pub const ASSIST_CLOSE: u32 = BYTES + 9;
pub const TOOL_OPEN: u32 = BYTES + 10;
pub const TOOL_CLOSE: u32 = BYTES + 11;
pub const RESULT_OPEN: u32 = BYTES + 12;
pub const RESULT_CLOSE: u32 = BYTES + 13;
pub const NEWLINE: u32 = 10;

pub fn base_vocab() -> u32 {
    BYTES + SPECIALS.len() as u32
}

pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>, // paire -> id de la fusion (sert aussi de priorite)
    vocab: Vec<Vec<u8>>,
    cache: Mutex<HashMap<Vec<u8>, Vec<u32>>>,
}

/// Decoupe un texte (sans jeton special) en morceaux sur lesquels le BPE s'applique.
fn pre_tokenize(s: &str) -> Vec<&[u8]> {
    let b = s.as_bytes();
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let end_of = |k: usize| if k + 1 < chars.len() { chars[k + 1].0 } else { b.len() };
    let mut out: Vec<&[u8]> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (start, c) = chars[i];
        if c.is_alphabetic() {
            let mut j = i;
            while j < chars.len() && chars[j].1.is_alphabetic() {
                j += 1;
            }
            out.push(&b[start..if j < chars.len() { chars[j].0 } else { b.len() }]);
            i = j;
        } else if c == ' ' {
            let mut j = i;
            while j < chars.len() && chars[j].1 == ' ' {
                j += 1;
            }
            if j < chars.len() && chars[j].1.is_alphabetic() {
                // le dernier espace de la serie s'attache au mot suivant
                if j - 1 > i {
                    out.push(&b[start..chars[j - 1].0]);
                }
                let mut k = j;
                while k < chars.len() && chars[k].1.is_alphabetic() {
                    k += 1;
                }
                out.push(&b[chars[j - 1].0..if k < chars.len() { chars[k].0 } else { b.len() }]);
                i = k;
            } else {
                out.push(&b[start..if j < chars.len() { chars[j].0 } else { b.len() }]);
                i = j;
            }
        } else {
            out.push(&b[start..end_of(i)]);
            i += 1;
        }
    }
    out
}

/// Separe un texte en segments : Ok(id) pour un jeton special, Err(texte) pour du texte ordinaire.
fn split_specials(text: &str) -> Vec<Result<u32, &str>> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut last = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'<' {
            let mut hit: Option<(usize, usize)> = None; // (id, longueur)
            for (k, sp) in SPECIALS.iter().enumerate() {
                if b[i..].starts_with(sp.as_bytes()) && hit.map_or(true, |(_, l)| sp.len() > l) {
                    hit = Some((k, sp.len()));
                }
            }
            if let Some((k, l)) = hit {
                if last < i {
                    out.push(Err(&text[last..i]));
                }
                out.push(Ok(BYTES + k as u32));
                i += l;
                last = i;
                continue;
            }
        }
        i += 1;
    }
    if last < b.len() {
        out.push(Err(&text[last..]));
    }
    out
}

impl Tokenizer {
    /// Tokenizer sans fusion : octets + jetons speciaux.
    pub fn new() -> Tokenizer {
        Tokenizer::from_merges(Vec::new()).expect("tokenizer vide")
    }

    pub fn from_merges(merges: Vec<(u32, u32)>) -> Result<Tokenizer, String> {
        let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for sp in SPECIALS {
            vocab.push(sp.as_bytes().to_vec());
        }
        let mut ranks = HashMap::new();
        for (i, &(a, b)) in merges.iter().enumerate() {
            let id = base_vocab() + i as u32;
            if a as usize >= vocab.len() || b as usize >= vocab.len() {
                return Err(format!("fusion {} invalide : ({}, {})", i, a, b));
            }
            let mut v = vocab[a as usize].clone();
            v.extend_from_slice(&vocab[b as usize]);
            vocab.push(v);
            ranks.insert((a, b), id);
        }
        Ok(Tokenizer { merges, ranks, vocab, cache: Mutex::new(HashMap::new()) })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Apprend `target_vocab - base` fusions sur un corpus (les jetons speciaux ne comptent pas).
    pub fn train(text: &str, target_vocab: usize) -> Tokenizer {
        let mut freq: HashMap<&[u8], usize> = HashMap::new();
        for seg in split_specials(text) {
            if let Err(t) = seg {
                for chunk in pre_tokenize(t) {
                    *freq.entry(chunk).or_insert(0) += 1;
                }
            }
        }
        // les chunks d'un seul caractere (chiffres, ponctuation) n'ont aucune paire a fusionner
        let mut words: Vec<(Vec<u32>, usize)> = freq
            .into_iter()
            .filter(|(w, _)| w.len() > 1)
            .map(|(w, c)| (w.iter().map(|&x| x as u32).collect(), c))
            .collect();
        words.sort(); // ordre deterministe
        let mut merges: Vec<(u32, u32)> = Vec::new();
        while base_vocab() as usize + merges.len() < target_vocab {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
            for (w, c) in &words {
                for p in w.windows(2) {
                    *counts.entry((p[0], p[1])).or_insert(0) += c;
                }
            }
            let best = counts.iter().filter(|(_, &c)| c >= 2).max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)));
            let (pair, _) = match best {
                Some((p, c)) => (*p, *c),
                None => break,
            };
            let new_id = base_vocab() + merges.len() as u32;
            merges.push(pair);
            for (w, _) in words.iter_mut() {
                merge_in_place(w, pair, new_id);
            }
        }
        Tokenizer::from_merges(merges).expect("fusions coherentes")
    }

    fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        if chunk.len() == 1 {
            return vec![chunk[0] as u32];
        }
        if let Ok(c) = self.cache.lock() {
            if let Some(v) = c.get(chunk) {
                return v.clone();
            }
        }
        let mut ids: Vec<u32> = chunk.iter().map(|&b| b as u32).collect();
        loop {
            let mut best: Option<(u32, (u32, u32))> = None;
            for p in ids.windows(2) {
                if let Some(&id) = self.ranks.get(&(p[0], p[1])) {
                    if best.map_or(true, |(bid, _)| id < bid) {
                        best = Some((id, (p[0], p[1])));
                    }
                }
            }
            match best {
                Some((id, pair)) => merge_in_place(&mut ids, pair, id),
                None => break,
            }
        }
        if let Ok(mut c) = self.cache.lock() {
            if c.len() < 50_000 {
                c.insert(chunk.to_vec(), ids.clone());
            }
        }
        ids
    }

    /// Texte -> ids. Les jetons speciaux ecrits en toutes lettres (ex. `<|user|>`) sont reconnus.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for seg in split_specials(text) {
            match seg {
                Ok(id) => out.push(id),
                Err(t) => {
                    for chunk in pre_tokenize(t) {
                        out.extend(self.encode_chunk(chunk));
                    }
                }
            }
        }
        out
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.vocab.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// ids -> texte (les jetons speciaux sont restitues tels quels, ex. `</|assistant|>`).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if id == PAD {
                continue;
            }
            bytes.extend_from_slice(self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Serialisation compacte des fusions : "a,b;c,d;..."
    pub fn merges_to_string(&self) -> String {
        self.merges.iter().map(|(a, b)| format!("{},{}", a, b)).collect::<Vec<_>>().join(";")
    }

    pub fn merges_from_string(s: &str) -> Result<Tokenizer, String> {
        let mut merges = Vec::new();
        for part in s.split(';').filter(|p| !p.is_empty()) {
            let mut it = part.split(',');
            let a = it.next().and_then(|x| x.parse::<u32>().ok()).ok_or("fusion illisible")?;
            let b = it.next().and_then(|x| x.parse::<u32>().ok()).ok_or("fusion illisible")?;
            merges.push((a, b));
        }
        Tokenizer::from_merges(merges)
    }
}

fn merge_in_place(w: &mut Vec<u32>, pair: (u32, u32), new_id: u32) {
    let mut i = 0;
    let mut o = 0;
    while i < w.len() {
        if i + 1 < w.len() && w[i] == pair.0 && w[i + 1] == pair.1 {
            w[o] = new_id;
            i += 2;
        } else {
            w[o] = w[i];
            i += 1;
        }
        o += 1;
    }
    w.truncate(o);
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "<|user|>\nBonjour, comment vas-tu ? J'aime bonjour et bonsoir.\n</|user|>\n<|assistant|>\nBonjour ! Bonjour bonjour 123\n</|assistant|>\n";

    #[test]
    fn aller_retour_exact() {
        let t = Tokenizer::train(&CORPUS.repeat(20), 300);
        for s in [CORPUS, "été à Kinshasa — çà et là", "  deux  espaces\t\ttab", "<|tools=\"python\">\nprint(1)\n</|tools>", ""] {
            assert_eq!(t.decode(&t.encode(s)), s);
        }
    }

    #[test]
    fn speciaux_et_chiffres() {
        let t = Tokenizer::train(&CORPUS.repeat(20), 300);
        let ids = t.encode("<|user|>x</|user|>");
        assert_eq!(ids[0], USER_OPEN);
        assert_eq!(*ids.last().unwrap(), USER_CLOSE);
        assert_eq!(t.encode("391").len(), 3);
        assert!(t.vocab_size() > base_vocab() as usize);
    }

    #[test]
    fn serialisation() {
        let t = Tokenizer::train(&CORPUS.repeat(20), 300);
        let t2 = Tokenizer::merges_from_string(&t.merges_to_string()).unwrap();
        assert_eq!(t.encode(CORPUS), t2.encode(CORPUS));
    }
}
