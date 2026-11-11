//! Orchestrateur : DAG de taches, sous-agents en parallele par vagues, reessais, annulation,
//! budget de taches, et un graphe de session partage qui sert de memoire.

use crate::bus::{Bus, Payload};
use crate::graph::{Kind, Rel, SessionGraph};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct Ctx {
    pub graph: Arc<Mutex<SessionGraph>>,
    pub bus: Arc<Bus>,
}

pub struct TaskInput {
    pub id: usize,
    pub name: String,
    pub prompt: String,
    /// Sous-graphe pertinent (texte borne) extrait du graphe de session.
    pub context: String,
    /// Sorties des taches dont celle-ci depend.
    pub deps_outputs: Vec<String>,
}

pub trait Agent: Send {
    fn handle(&mut self, input: &TaskInput, ctx: &Ctx) -> Result<String, String>;
}

/// Agent construit a partir d'une closure.
pub struct FnAgent<F>(pub F);

impl<F> FnAgent<F>
where
    F: FnMut(&TaskInput, &Ctx) -> Result<String, String> + Send,
{
    pub fn new(f: F) -> FnAgent<F> {
        FnAgent(f)
    }
}

impl<F> Agent for FnAgent<F>
where
    F: FnMut(&TaskInput, &Ctx) -> Result<String, String> + Send,
{
    fn handle(&mut self, input: &TaskInput, ctx: &Ctx) -> Result<String, String> {
        (self.0)(input, ctx)
    }
}

/// Interroge le graphe de session avec le prompt de la tache.
pub struct RecallAgent;

impl Agent for RecallAgent {
    fn handle(&mut self, input: &TaskInput, ctx: &Ctx) -> Result<String, String> {
        let mut g = ctx.graph.lock().map_err(|_| "verrou du graphe empoisonne".to_string())?;
        Ok(g.recall(&input.prompt, 5, 2, 1500))
    }
}

/// Resume extractif simple : premiere phrase de chaque entree.
pub struct SummarizeAgent;

impl Agent for SummarizeAgent {
    fn handle(&mut self, input: &TaskInput, _ctx: &Ctx) -> Result<String, String> {
        let mut parts: Vec<String> = Vec::new();
        for text in input.deps_outputs.iter().chain(std::iter::once(&input.context)) {
            for line in text.lines() {
                let l = line.trim();
                if l.is_empty() {
                    continue;
                }
                let first = l.split('.').next().unwrap_or(l).trim();
                if !first.is_empty() {
                    parts.push(first.to_string());
                }
            }
        }
        parts.dedup();
        Ok(parts.join(". "))
    }
}

#[derive(Clone, Debug)]
pub struct TaskSpec {
    pub name: String,
    pub agent: String,
    pub prompt: String,
    /// Indices de taches precedentes (strictement inferieurs a l'indice courant : pas de cycle possible).
    pub deps: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct TaskResult {
    pub id: usize,
    pub name: String,
    pub agent: String,
    pub ok: bool,
    pub output: String,
    pub attempts: u32,
}

pub struct Orchestrator {
    agents: HashMap<String, Arc<Mutex<Box<dyn Agent>>>>,
    pub graph: Arc<Mutex<SessionGraph>>,
    pub bus: Arc<Bus>,
    pub max_retries: u32,
    pub max_tasks: usize,
    pub cancel: Arc<AtomicBool>,
}

impl Orchestrator {
    pub fn new(session: &str) -> Orchestrator {
        Orchestrator {
            agents: HashMap::new(),
            graph: Arc::new(Mutex::new(SessionGraph::new(session))),
            bus: Arc::new(Bus::new()),
            max_retries: 1,
            max_tasks: 256,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn register(&mut self, name: &str, agent: Box<dyn Agent>) {
        self.agents.insert(name.to_string(), Arc::new(Mutex::new(agent)));
    }

    pub fn run(&self, tasks: &[TaskSpec]) -> Result<Vec<TaskResult>, String> {
        if tasks.len() > self.max_tasks {
            return Err(format!("budget depasse : {} taches (max {})", tasks.len(), self.max_tasks));
        }
        for (i, t) in tasks.iter().enumerate() {
            if !self.agents.contains_key(&t.agent) {
                return Err(format!("tache {} : agent inconnu '{}'", i, t.agent));
            }
            for &d in &t.deps {
                if d >= i {
                    return Err(format!("tache {} : la dependance {} doit preceder la tache", i, d));
                }
            }
        }
        let n = tasks.len();
        let mut results: Vec<Option<TaskResult>> = vec![None; n];
        let mut node_ids: Vec<u64> = Vec::new();
        {
            let mut g = self.graph.lock().map_err(|_| "verrou du graphe empoisonne".to_string())?;
            for (i, t) in tasks.iter().enumerate() {
                let id = g.add_node(Kind::Task, &format!("{} : {}", t.name, t.prompt));
                node_ids.push(id);
                for &d in &t.deps {
                    let _ = g.add_edge(id, node_ids[d], Rel::DependsOn, 1.0);
                }
                let _ = i;
            }
        }
        let mut done = 0usize;
        while done < n {
            if self.cancel.load(Ordering::SeqCst) {
                return Err("execution annulee".to_string());
            }
            let mut ready: Vec<usize> = Vec::new();
            let mut progress = false;
            for i in 0..n {
                if results[i].is_some() {
                    continue;
                }
                let mut all_ok = true;
                let mut failed_dep = false;
                for &d in &tasks[i].deps {
                    match &results[d] {
                        Some(r) if r.ok => {}
                        Some(_) => {
                            failed_dep = true;
                            all_ok = false;
                        }
                        None => all_ok = false,
                    }
                }
                if failed_dep {
                    results[i] = Some(TaskResult {
                        id: i,
                        name: tasks[i].name.clone(),
                        agent: tasks[i].agent.clone(),
                        ok: false,
                        output: "dependance echouee".to_string(),
                        attempts: 0,
                    });
                    done += 1;
                    progress = true;
                    continue;
                }
                if all_ok {
                    ready.push(i);
                }
            }
            if ready.is_empty() {
                if progress {
                    continue;
                }
                return Err("blocage de l'ordonnanceur".to_string());
            }
            let mut inputs: Vec<TaskInput> = Vec::new();
            for &i in &ready {
                let context = match self.graph.lock() {
                    Ok(mut g) => g.recall(&tasks[i].prompt, 5, 2, 1200),
                    Err(_) => String::new(),
                };
                let deps_outputs: Vec<String> = tasks[i]
                    .deps
                    .iter()
                    .filter_map(|&d| results[d].as_ref().map(|r| r.output.clone()))
                    .collect();
                inputs.push(TaskInput {
                    id: i,
                    name: tasks[i].name.clone(),
                    prompt: tasks[i].prompt.clone(),
                    context,
                    deps_outputs,
                });
            }
            let outcomes: Vec<(usize, Result<String, String>, u32)> = std::thread::scope(|s| {
                let handles: Vec<(usize, _)> = inputs
                    .iter()
                    .map(|inp| {
                        let agent = self.agents[&tasks[inp.id].agent].clone();
                        let ctx = Ctx { graph: self.graph.clone(), bus: self.bus.clone() };
                        let retries = self.max_retries;
                        let h = s.spawn(move || {
                            let mut attempts = 0u32;
                            let mut last: Result<String, String> = Err("aucune tentative".to_string());
                            while attempts <= retries {
                                attempts += 1;
                                let r = match agent.lock() {
                                    Ok(mut a) => a.handle(inp, &ctx),
                                    Err(_) => Err("agent indisponible (panique precedente)".to_string()),
                                };
                                let ok = r.is_ok();
                                last = r;
                                if ok {
                                    break;
                                }
                            }
                            (inp.id, last, attempts)
                        });
                        (inp.id, h)
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|(id, h)| match h.join() {
                        Ok(r) => r,
                        Err(_) => (id, Err("panique de l'agent".to_string()), 0),
                    })
                    .collect()
            });
            for (id, res, attempts) in outcomes {
                let (ok, output) = match res {
                    Ok(o) => (true, o),
                    Err(e) => (false, e),
                };
                results[id] = Some(TaskResult {
                    id,
                    name: tasks[id].name.clone(),
                    agent: tasks[id].agent.clone(),
                    ok,
                    output: output.clone(),
                    attempts,
                });
                done += 1;
                if let Ok(mut g) = self.graph.lock() {
                    let (kind, text) = if ok {
                        (Kind::Fact, format!("{} : {}", tasks[id].name, output))
                    } else {
                        (Kind::Message, format!("ECHEC {} : {}", tasks[id].name, output))
                    };
                    let nid = g.add_node(kind, &text);
                    let _ = g.add_edge(nid, node_ids[id], Rel::ProducedBy, 1.0);
                    let _ = g.compact(200);
                }
                self.bus.publish(
                    "task.done",
                    &tasks[id].agent,
                    Payload::Text(format!("{}:{}", tasks[id].name, if ok { "ok" } else { "echec" })),
                );
            }
        }
        Ok(results.into_iter().flatten().collect())
    }
}
