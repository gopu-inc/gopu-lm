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

/// Pre-decoupage du texte avant le BPE.
/// `Legacy` : celui des modeles gopu-lm existants. `Gpt` : equivalent exact de l'expression reguliere
/// ` ?\p{L}+|\p{N}|[^\s\p{L}\p{N}]|\s+(?!\S)|\s+` (utilisable telle quelle dans tokenizer.json de Hugging Face).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PreTok {
    Legacy,
    Gpt,
}

impl PreTok {
    pub fn name(&self) -> &'static str {
        match self {
            PreTok::Legacy => "legacy",
            PreTok::Gpt => "gpt",
        }
    }
    pub fn from_name(s: &str) -> PreTok {
        if s == "gpt" {
            PreTok::Gpt
        } else {
            PreTok::Legacy
        }
    }
}

pub const GPT_SPLIT_REGEX: &str = " ?\\p{L}+|\\p{N}|[^\\s\\p{L}\\p{N}]|\\s+(?!\\S)|\\s+";

pub struct Tokenizer {
    pretok: PreTok,
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

/// Pre-decoupage "Gpt" : voir `PreTok::Gpt`.
fn pre_tokenize_gpt(s: &str) -> Vec<&[u8]> {
    let b = s.as_bytes();
    let ch: Vec<(usize, char)> = s.char_indices().collect();
    let n = ch.len();
    let end = |k: usize| if k < n { ch[k].0 } else { b.len() };
    let mut out: Vec<&[u8]> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = ch[i].1;
        let start = ch[i].0;
        if c.is_alphabetic() {
            let mut j = i;
            while j < n && ch[j].1.is_alphabetic() {
                j += 1;
            }
            out.push(&b[start..end(j)]);
            i = j;
        } else if c == ' ' && i + 1 < n && ch[i + 1].1.is_alphabetic() {
            let mut j = i + 1;
            while j < n && ch[j].1.is_alphabetic() {
                j += 1;
            }
            out.push(&b[start..end(j)]);
            i = j;
        } else if c.is_numeric() || !c.is_whitespace() {
            out.push(&b[start..end(i + 1)]);
            i += 1;
        } else {
            let mut j = i;
            while j < n && ch[j].1.is_whitespace() {
                j += 1;
            }
            if j == n || j - i == 1 {
                out.push(&b[start..end(j)]);
                i = j;
            } else {
                out.push(&b[start..end(j - 1)]);
                i = j - 1;
            }
        }
    }
    out
}

fn pre_tokenize_mode(s: &str, m: PreTok) -> Vec<&[u8]> {
    match m {
        PreTok::Legacy => pre_tokenize(s),
        PreTok::Gpt => pre_tokenize_gpt(s),
    }
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
        Tokenizer::from_merges_mode(merges, PreTok::Legacy)
    }

    pub fn pretok(&self) -> PreTok {
        self.pretok
    }

    pub fn from_merges_mode(merges: Vec<(u32, u32)>, pretok: PreTok) -> Result<Tokenizer, String> {
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
        Ok(Tokenizer { pretok, merges, ranks, vocab, cache: Mutex::new(HashMap::new()) })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Apprend jusqu'a `target_vocab - base` fusions sur un corpus (pre-decoupage `Legacy`).
    pub fn train(text: &str, target_vocab: usize) -> Tokenizer {
        Tokenizer::train_with(text, target_vocab, PreTok::Legacy)
    }

    /// BPE incremental (les compteurs de paires ne sont mis a jour que sur les mots touches).
    /// Une fusion qui reproduirait une chaine d'octets deja presente dans le vocabulaire est ecartee
    /// (le vocabulaire Hugging Face exige des chaines uniques).
    pub fn train_with(text: &str, target_vocab: usize, pretok: PreTok) -> Tokenizer {
        let mut freq: HashMap<&[u8], usize> = HashMap::new();
        for seg in split_specials(text) {
            if let Err(t) = seg {
                for chunk in pre_tokenize_mode(t, pretok) {
                    *freq.entry(chunk).or_insert(0) += 1;
                }
            }
        }
        let mut words: Vec<(Vec<u32>, usize)> = freq
            .into_iter()
            .filter(|(w, _)| w.len() > 1)
            .map(|(w, c)| (w.iter().map(|&x| x as u32).collect(), c))
            .collect();
        words.sort();
        let mut counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for (wi, (w, c)) in words.iter().enumerate() {
            for p in w.windows(2) {
                *counts.entry((p[0], p[1])).or_insert(0) += *c as i64;
                where_.entry((p[0], p[1])).or_default().push(wi);
            }
        }
        let mut vocab_bytes: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for sp in SPECIALS {
            vocab_bytes.push(sp.as_bytes().to_vec());
        }
        let mut seen: std::collections::HashSet<Vec<u8>> = vocab_bytes.iter().cloned().collect();
        let mut banned: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut merges: Vec<(u32, u32)> = Vec::new();
        while vocab_bytes.len() < target_vocab {
            let best = counts.iter().filter(|(p, &c)| c >= 2 && !banned.contains(*p)).max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0))).map(|(p, _)| *p);
            let pair = match best {
                Some(p) => p,
                None => break,
            };
            let mut bytes = vocab_bytes[pair.0 as usize].clone();
            bytes.extend_from_slice(&vocab_bytes[pair.1 as usize]);
            if seen.contains(&bytes) {
                banned.insert(pair);
                continue;
            }
            let new_id = vocab_bytes.len() as u32;
            seen.insert(bytes.clone());
            vocab_bytes.push(bytes);
            merges.push(pair);
            let mut idxs = where_.remove(&pair).unwrap_or_default();
            idxs.sort_unstable();
            idxs.dedup();
            for wi in idxs {
                let cnt = words[wi].1 as i64;
                if !words[wi].0.windows(2).any(|p| (p[0], p[1]) == pair) {
                    continue;
                }
                for p in words[wi].0.windows(2) {
                    if let Some(c) = counts.get_mut(&(p[0], p[1])) {
                        *c -= cnt;
                    }
                }
                merge_in_place(&mut words[wi].0, pair, new_id);
                for p in words[wi].0.windows(2) {
                    *counts.entry((p[0], p[1])).or_insert(0) += cnt;
                    where_.entry((p[0], p[1])).or_default().push(wi);
                }
            }
            counts.remove(&pair);
        }
        Tokenizer::from_merges_mode(merges, pretok).expect("fusions coherentes")
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
                    for chunk in pre_tokenize_mode(t, self.pretok) {
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

// ------------------------------------------------------------------ export Hugging Face (tokenizers, BPE au niveau des octets)

/// Table octet -> caractere visible de GPT-2 (`bytes_to_unicode`).
pub fn byte_unicode() -> Vec<char> {
    let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut map = vec![' '; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        map[*b as usize] = char::from_u32(*c).unwrap();
    }
    map
}

impl Tokenizer {
    /// Representation Hugging Face d'un jeton non special (octets -> caracteres visibles).
    pub fn hf_token(&self, id: u32) -> String {
        let m = byte_unicode();
        self.token_bytes(id).iter().map(|&b| m[b as usize]).collect()
    }

    /// vocab.json : { "jeton": id } (jetons speciaux inclus, ids 256..).
    pub fn hf_vocab_json(&self) -> String {
        use crate::json::{self, Value};
        let mut o: Vec<(String, Value)> = Vec::new();
        for id in 0..self.vocab.len() as u32 {
            let name = if (BYTES..base_vocab()).contains(&id) { SPECIALS[(id - BYTES) as usize].to_string() } else { self.hf_token(id) };
            o.push((name, Value::Num(id as f64)));
        }
        json::to_string(&Value::Obj(o))
    }

    /// merges.txt (une fusion par ligne, dans l'ordre de priorite).
    pub fn hf_merges_txt(&self) -> String {
        let mut s = String::from("#version: 0.2\n");
        for &(a, b) in &self.merges {
            s.push_str(&format!("{} {}\n", self.hf_token(a), self.hf_token(b)));
        }
        s
    }

    /// tokenizer.json (format de la bibliotheque `tokenizers`).
    pub fn hf_tokenizer_json(&self) -> String {
        use crate::json::{self, Value};
        let st = |x: &str| Value::Str(x.to_string());
        let added: Vec<Value> = SPECIALS
            .iter()
            .enumerate()
            .map(|(k, sp)| {
                Value::Obj(vec![
                    ("id".into(), Value::Num((BYTES as usize + k) as f64)),
                    ("content".into(), st(sp)),
                    ("single_word".into(), Value::Bool(false)),
                    ("lstrip".into(), Value::Bool(false)),
                    ("rstrip".into(), Value::Bool(false)),
                    ("normalized".into(), Value::Bool(false)),
                    ("special".into(), Value::Bool(true)),
                ])
            })
            .collect();
        let mut vocab: Vec<(String, Value)> = Vec::new();
        for id in 0..self.vocab.len() as u32 {
            if !(BYTES..base_vocab()).contains(&id) {
                vocab.push((self.hf_token(id), Value::Num(id as f64)));
            }
        }
        let merges: Vec<Value> = self.merges.iter().map(|&(a, b)| Value::Str(format!("{} {}", self.hf_token(a), self.hf_token(b)))).collect();
        let pre = if self.pretok == PreTok::Gpt {
            Value::Obj(vec![
                ("type".into(), st("Sequence")),
                (
                    "pretokenizers".into(),
                    Value::Arr(vec![
                        Value::Obj(vec![
                            ("type".into(), st("Split")),
                            ("pattern".into(), Value::Obj(vec![("Regex".into(), st(GPT_SPLIT_REGEX))])),
                            ("behavior".into(), st("Isolated")),
                            ("invert".into(), Value::Bool(false)),
                        ]),
                        Value::Obj(vec![("type".into(), st("ByteLevel")), ("add_prefix_space".into(), Value::Bool(false)), ("trim_offsets".into(), Value::Bool(true)), ("use_regex".into(), Value::Bool(false))]),
                    ]),
                ),
            ])
        } else {
            Value::Null
        };
        let model = Value::Obj(vec![
            ("type".into(), st("BPE")),
            ("dropout".into(), Value::Null),
            ("unk_token".into(), Value::Null),
            ("continuing_subword_prefix".into(), Value::Null),
            ("end_of_word_suffix".into(), Value::Null),
            ("fuse_unk".into(), Value::Bool(false)),
            ("byte_fallback".into(), Value::Bool(false)),
            ("ignore_merges".into(), Value::Bool(false)),
            ("vocab".into(), Value::Obj(vocab)),
            ("merges".into(), Value::Arr(merges)),
        ]);
        json::to_string(&Value::Obj(vec![
            ("version".into(), st("1.0")),
            ("truncation".into(), Value::Null),
            ("padding".into(), Value::Null),
            ("added_tokens".into(), Value::Arr(added)),
            ("normalizer".into(), Value::Null),
            ("pre_tokenizer".into(), pre),
            ("post_processor".into(), Value::Null),
            ("decoder".into(), Value::Obj(vec![("type".into(), st("ByteLevel")), ("add_prefix_space".into(), Value::Bool(true)), ("trim_offsets".into(), Value::Bool(true)), ("use_regex".into(), Value::Bool(true))])),
            ("model".into(), model),
        ]))
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
    fn mode_gpt_decoupage_et_aller_retour() {
        let t = Tokenizer::train_with(&CORPUS.repeat(30), 320, PreTok::Gpt);
        for s in ["Bonjour  monde\n\nfin  ", " a  b\t\tc", "été (ok) 12,5 ?\n", "  ", "x"] {
            assert_eq!(t.decode(&t.encode(s)), s);
        }
        let p = pre_tokenize_gpt("ab  cd\n\nef 12");
        let v: Vec<&str> = p.iter().map(|x| std::str::from_utf8(x).unwrap()).collect();
        assert_eq!(v, vec!["ab", " ", " cd", "\n", "\n", "ef", " ", "1", "2"].iter().map(|x| *x).collect::<Vec<_>>().as_slice() as &[&str]);
        // les chaines du vocabulaire HF sont uniques
        let j = t.hf_vocab_json();
        let n = crate::json::parse(&j).unwrap();
        match n {
            crate::json::Value::Obj(o) => assert_eq!(o.len(), t.vocab_size()),
            _ => panic!("objet attendu"),
        }
    }

    #[test]
    fn serialisation() {
        let t = Tokenizer::train(&CORPUS.repeat(20), 300);
        let t2 = Tokenizer::merges_from_string(&t.merges_to_string()).unwrap();
        assert_eq!(t.encode(CORPUS), t2.encode(CORPUS));
    }
}
