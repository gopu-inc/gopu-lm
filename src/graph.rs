//! Graphe de session : une session = un graphe qui conserve toutes les connaissances.
//! Les agents echangent des identifiants de noeuds et demandent un sous-graphe borne (recall).
//!
//! Note : l'embedding est un "hashing trick" lexical (sac de mots), pas un embedding semantique appris.

use std::collections::HashMap;
use std::fs;

pub const DIM: usize = 128;

pub fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    let lower = text.to_lowercase();
    for tok in lower.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in tok.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let idx = (h % DIM as u64) as usize;
        let sign = if (h >> 33) & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut o: String = s.chars().take(max).collect();
        o.push_str("...");
        o
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Message,
    Fact,
    Entity,
    Task,
    Summary,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Message => "message",
            Kind::Fact => "fact",
            Kind::Entity => "entity",
            Kind::Task => "task",
            Kind::Summary => "summary",
        }
    }
    pub fn parse(s: &str) -> Option<Kind> {
        match s {
            "message" => Some(Kind::Message),
            "fact" => Some(Kind::Fact),
            "entity" => Some(Kind::Entity),
            "task" => Some(Kind::Task),
            "summary" => Some(Kind::Summary),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rel {
    Mentions,
    DependsOn,
    ProducedBy,
    Contradicts,
    Follows,
    Supersedes,
}

impl Rel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Rel::Mentions => "mentions",
            Rel::DependsOn => "depends_on",
            Rel::ProducedBy => "produced_by",
            Rel::Contradicts => "contradicts",
            Rel::Follows => "follows",
            Rel::Supersedes => "supersedes",
        }
    }
    pub fn parse(s: &str) -> Option<Rel> {
        match s {
            "mentions" => Some(Rel::Mentions),
            "depends_on" => Some(Rel::DependsOn),
            "produced_by" => Some(Rel::ProducedBy),
            "contradicts" => Some(Rel::Contradicts),
            "follows" => Some(Rel::Follows),
            "supersedes" => Some(Rel::Supersedes),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: u64,
    pub kind: Kind,
    pub emb: Vec<f32>,
    pub text: String,
    pub created: u64,
    pub last_access: u64,
    pub weight: f32,
    pub archived: bool,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: u64,
    pub to: u64,
    pub rel: Rel,
    pub weight: f32,
}

pub struct SessionGraph {
    pub session: String,
    nodes: HashMap<u64, Node>,
    edges: Vec<Edge>,
    next_id: u64,
    clock: u64,
}

fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\t' => o.push_str("\\t"),
            '\r' => o.push_str("\\r"),
            _ => o.push(c),
        }
    }
    o
}

fn unesc(s: &str) -> String {
    let mut o = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => o.push('\n'),
                Some('t') => o.push('\t'),
                Some('r') => o.push('\r'),
                Some('\\') => o.push('\\'),
                Some(x) => o.push(x),
                None => {}
            }
        } else {
            o.push(c);
        }
    }
    o
}

fn pnum<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
    s.parse::<T>().map_err(|_| format!("valeur invalide pour {} : '{}'", what, s))
}

impl SessionGraph {
    pub fn new(session: &str) -> SessionGraph {
        SessionGraph { session: session.to_string(), nodes: HashMap::new(), edges: Vec::new(), next_id: 1, clock: 0 }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    pub fn add_node(&mut self, kind: Kind, text: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let t = self.tick();
        self.nodes.insert(
            id,
            Node { id, kind, emb: embed(text), text: text.to_string(), created: t, last_access: t, weight: 1.0, archived: false },
        );
        id
    }

    pub fn add_edge(&mut self, from: u64, to: u64, rel: Rel, weight: f32) -> Result<(), String> {
        if !self.nodes.contains_key(&from) || !self.nodes.contains_key(&to) {
            return Err(format!("noeud inconnu dans l'arete {} -> {}", from, to));
        }
        self.edges.push(Edge { from, to, rel, weight });
        Ok(())
    }

    pub fn node(&self, id: u64) -> Option<&Node> {
        self.nodes.get(&id)
    }
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn active_len(&self) -> usize {
        self.nodes.values().filter(|n| !n.archived).count()
    }
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Points d'entree : noeuds actifs les plus proches de la requete.
    pub fn search(&self, query: &str, k: usize) -> Vec<(u64, f32)> {
        let q = embed(query);
        let mut r: Vec<(u64, f32)> = self
            .nodes
            .values()
            .filter(|n| !n.archived)
            .map(|n| (n.id, dot(&q, &n.emb)))
            .filter(|(_, s)| *s > 0.0)
            .collect();
        r.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        r.truncate(k);
        r
    }

    /// Sous-graphe pertinent serialise en texte, borne par `budget_chars`.
    /// Propagation d'activation : l'activation decroit a chaque saut (poids de l'arete x 0.5).
    pub fn recall(&mut self, query: &str, k: usize, depth: usize, budget_chars: usize) -> String {
        let seeds = self.search(query, k);
        let mut adj: HashMap<u64, Vec<(u64, f32)>> = HashMap::new();
        for e in &self.edges {
            adj.entry(e.from).or_default().push((e.to, e.weight));
            adj.entry(e.to).or_default().push((e.from, e.weight));
        }
        let mut act: HashMap<u64, f32> = HashMap::new();
        let mut frontier: Vec<(u64, f32)> = Vec::new();
        for (id, s) in seeds {
            act.insert(id, s);
            frontier.push((id, s));
        }
        for _ in 0..depth {
            let mut next: Vec<(u64, f32)> = Vec::new();
            for (id, a) in &frontier {
                if let Some(neigh) = adj.get(id) {
                    for (o, w) in neigh {
                        let na = a * w * 0.5;
                        if na < 0.05 {
                            continue;
                        }
                        let alive = self.nodes.get(o).map(|n| !n.archived).unwrap_or(false);
                        if !alive {
                            continue;
                        }
                        let cur = act.get(o).copied().unwrap_or(0.0);
                        if na > cur {
                            act.insert(*o, na);
                            next.push((*o, na));
                        }
                    }
                }
            }
            frontier = next;
        }
        let mut ranked: Vec<(u64, f32)> = act.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        let now = self.tick();
        let mut out = String::new();
        for (id, _) in ranked {
            if let Some(n) = self.nodes.get_mut(&id) {
                let line = format!("[{}#{}] {}\n", n.kind.as_str(), n.id, clip(&n.text, 400));
                if out.len() + line.len() > budget_chars && !out.is_empty() {
                    break;
                }
                out.push_str(&line);
                n.last_access = now;
                n.weight = (n.weight + 0.1).min(2.0);
            }
        }
        out
    }

    /// Compaction : les plus vieux messages actifs au-dela de `max_active_messages` sont remplaces par un resume.
    /// Les originaux restent dans le graphe (archives, relies par `supersedes`).
    pub fn compact(&mut self, max_active_messages: usize) -> Option<u64> {
        let mut msgs: Vec<u64> = self
            .nodes
            .values()
            .filter(|n| n.kind == Kind::Message && !n.archived)
            .map(|n| n.id)
            .collect();
        msgs.sort();
        if msgs.len() <= max_active_messages {
            return None;
        }
        let excess = msgs.len() - max_active_messages;
        let old: Vec<u64> = msgs.into_iter().take(excess).collect();
        let mut text = String::from("Resume : ");
        for id in &old {
            if let Some(n) = self.nodes.get(id) {
                text.push_str(&clip(&n.text, 80));
                text.push_str(" | ");
            }
        }
        let sid = self.add_node(Kind::Summary, &text);
        for id in &old {
            if let Some(n) = self.nodes.get_mut(id) {
                n.archived = true;
            }
            self.edges.push(Edge { from: sid, to: *id, rel: Rel::Supersedes, weight: 1.0 });
        }
        Some(sid)
    }

    /// Oubli pondere : le poids decroit, sous `floor` le noeud est archive (jamais supprime). Retourne le nombre archive.
    pub fn decay(&mut self, factor: f32, floor: f32) -> usize {
        let mut c = 0;
        for n in self.nodes.values_mut() {
            if n.archived || n.kind == Kind::Summary {
                continue;
            }
            n.weight *= factor;
            if n.weight < floor {
                n.archived = true;
                c += 1;
            }
        }
        c
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut s = String::new();
        s.push_str(&format!("S\t{}\t{}\t{}\n", esc(&self.session), self.next_id, self.clock));
        let mut ids: Vec<&u64> = self.nodes.keys().collect();
        ids.sort();
        for id in ids {
            let n = &self.nodes[id];
            s.push_str(&format!(
                "N\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                n.id,
                n.kind.as_str(),
                n.created,
                n.last_access,
                n.weight,
                n.archived as u8,
                esc(&n.text)
            ));
        }
        for e in &self.edges {
            s.push_str(&format!("E\t{}\t{}\t{}\t{}\n", e.from, e.to, e.rel.as_str(), e.weight));
        }
        let tmp = format!("{}.tmp", path);
        fs::write(&tmp, s).map_err(|e| e.to_string())?;
        fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    pub fn load(path: &str) -> Result<SessionGraph, String> {
        let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut g = SessionGraph::new("");
        for (ln, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let p: Vec<&str> = line.split('\t').collect();
            match p[0] {
                "S" => {
                    if p.len() < 4 {
                        return Err(format!("ligne {} invalide", ln + 1));
                    }
                    g.session = unesc(p[1]);
                    g.next_id = pnum(p[2], "next_id")?;
                    g.clock = pnum(p[3], "clock")?;
                }
                "N" => {
                    if p.len() < 8 {
                        return Err(format!("ligne {} invalide", ln + 1));
                    }
                    let id: u64 = pnum(p[1], "id")?;
                    let kind = Kind::parse(p[2]).ok_or_else(|| format!("type inconnu '{}'", p[2]))?;
                    let created: u64 = pnum(p[3], "created")?;
                    let last_access: u64 = pnum(p[4], "last_access")?;
                    let weight: f32 = pnum(p[5], "weight")?;
                    let archived = p[6] == "1";
                    let text = unesc(p[7]);
                    let emb = embed(&text);
                    g.nodes.insert(id, Node { id, kind, emb, text, created, last_access, weight, archived });
                }
                "E" => {
                    if p.len() < 5 {
                        return Err(format!("ligne {} invalide", ln + 1));
                    }
                    let from: u64 = pnum(p[1], "from")?;
                    let to: u64 = pnum(p[2], "to")?;
                    let rel = Rel::parse(p[3]).ok_or_else(|| format!("relation inconnue '{}'", p[3]))?;
                    let weight: f32 = pnum(p[4], "weight")?;
                    g.edges.push(Edge { from, to, rel, weight });
                }
                other => return Err(format!("ligne {} : balise '{}' inconnue", ln + 1, other)),
            }
        }
        Ok(g)
    }
}
