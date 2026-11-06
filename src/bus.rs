//! Bus de messages : boites aux lettres par agent + pub/sub par topic. std uniquement.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

#[derive(Clone, Debug)]
pub enum Payload {
    Text(String),
    Bytes(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct Envelope {
    pub id: u64,
    pub corr_id: Option<u64>,
    pub from: String,
    pub to: String,
    pub topic: Option<String>,
    pub payload: Payload,
}

#[derive(Default)]
struct Inner {
    boxes: HashMap<String, VecDeque<Envelope>>,
    subs: HashMap<String, Vec<String>>,
}

pub struct Bus {
    inner: Mutex<Inner>,
    next: AtomicU64,
}

impl Default for Bus {
    fn default() -> Self {
        Bus::new()
    }
}

impl Bus {
    pub fn new() -> Bus {
        Bus { inner: Mutex::new(Inner::default()), next: AtomicU64::new(1) }
    }

    fn next_id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::SeqCst)
    }

    /// Message direct. `corr_id` relie une reponse a sa requete.
    pub fn send(&self, from: &str, to: &str, corr_id: Option<u64>, payload: Payload) -> u64 {
        let id = self.next_id();
        let env = Envelope { id, corr_id, from: from.to_string(), to: to.to_string(), topic: None, payload };
        if let Ok(mut g) = self.inner.lock() {
            g.boxes.entry(to.to_string()).or_default().push_back(env);
        }
        id
    }

    pub fn subscribe(&self, agent: &str, topic: &str) {
        if let Ok(mut g) = self.inner.lock() {
            let list = g.subs.entry(topic.to_string()).or_default();
            if !list.iter().any(|a| a == agent) {
                list.push(agent.to_string());
            }
        }
    }

    /// Diffuse a tous les abonnes du topic. Retourne le nombre de destinataires.
    pub fn publish(&self, topic: &str, from: &str, payload: Payload) -> usize {
        let id = self.next_id();
        let mut n = 0;
        if let Ok(mut g) = self.inner.lock() {
            let subs = g.subs.get(topic).cloned().unwrap_or_default();
            for s in subs {
                let env = Envelope {
                    id,
                    corr_id: None,
                    from: from.to_string(),
                    to: s.clone(),
                    topic: Some(topic.to_string()),
                    payload: payload.clone(),
                };
                g.boxes.entry(s).or_default().push_back(env);
                n += 1;
            }
        }
        n
    }

    pub fn recv(&self, agent: &str) -> Option<Envelope> {
        let mut g = self.inner.lock().ok()?;
        let r = g.boxes.get_mut(agent)?.pop_front();
        r
    }

    pub fn pending(&self, agent: &str) -> usize {
        match self.inner.lock() {
            Ok(g) => g.boxes.get(agent).map(|b| b.len()).unwrap_or(0),
            Err(_) => 0,
        }
    }
}
