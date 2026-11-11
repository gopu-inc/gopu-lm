//! Interface C (cdylib) utilisee par le paquet Python via ctypes : zero dependance cote Python.

use crate::graph::{Kind, SessionGraph};
use crate::json::{self, Value};
use crate::nn::{Act, Mlp};
use crate::orch::{Agent, Ctx, Orchestrator, RecallAgent, SummarizeAgent, TaskInput, TaskSpec};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;

thread_local! {
    static LAST_ERR: RefCell<String> = RefCell::new(String::new());
}

pub(crate) fn set_err(e: String) {
    LAST_ERR.with(|c| *c.borrow_mut() = e);
}

pub(crate) unsafe fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

pub(crate) fn to_c(s: String) -> *mut c_char {
    match CString::new(s.replace('\0', " ")) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn gopu_version() -> *const c_char {
    b"gopu 0.1.0\0".as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn gopu_last_error() -> *mut c_char {
    let s = LAST_ERR.with(|c| c.borrow().clone());
    to_c(s)
}

#[no_mangle]
pub unsafe extern "C" fn gopu_string_free(p: *mut c_char) {
    if !p.is_null() {
        drop(CString::from_raw(p));
    }
}

// ---------------------------------------------------------------- modele

#[no_mangle]
pub unsafe extern "C" fn gopu_model_new(dims: *const u32, n: usize, acts: *const c_char, seed: u64) -> *mut Mlp {
    if dims.is_null() || n < 2 {
        set_err("dims invalides".to_string());
        return std::ptr::null_mut();
    }
    let d: Vec<usize> = std::slice::from_raw_parts(dims, n).iter().map(|&x| x as usize).collect();
    let acts_s = cstr(acts);
    let parsed: Result<Vec<Act>, String> = acts_s.split(',').map(|s| Act::parse(s.trim())).collect();
    match parsed.and_then(|a| Mlp::new(&d, &a, seed)) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_model_free(m: *mut Mlp) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_model_in_dim(m: *const Mlp) -> i32 {
    if m.is_null() {
        return -1;
    }
    (*m).in_dim() as i32
}

#[no_mangle]
pub unsafe extern "C" fn gopu_model_out_dim(m: *const Mlp) -> i32 {
    if m.is_null() {
        return -1;
    }
    (*m).out_dim() as i32
}

/// Entraine le modele. Retourne la derniere perte moyenne, ou NaN en cas d'erreur (voir gopu_last_error).
#[no_mangle]
pub unsafe extern "C" fn gopu_model_train(
    m: *mut Mlp,
    x: *const f32,
    rows: usize,
    cols: usize,
    labels: *const u32,
    epochs: u32,
    lr: f32,
    batch: usize,
) -> f32 {
    if m.is_null() || x.is_null() || labels.is_null() {
        set_err("pointeur nul".to_string());
        return f32::NAN;
    }
    let xs = match Tensor::from_vec(std::slice::from_raw_parts(x, rows * cols).to_vec(), &[rows, cols]) {
        Ok(t) => t,
        Err(e) => {
            set_err(e);
            return f32::NAN;
        }
    };
    let lab: Vec<usize> = std::slice::from_raw_parts(labels, rows).iter().map(|&l| l as usize).collect();
    match (*m).fit(&xs, &lab, epochs as usize, lr, batch) {
        Ok(h) => h.last().copied().unwrap_or(f32::NAN),
        Err(e) => {
            set_err(e);
            f32::NAN
        }
    }
}

/// Ecrit rows * out_dim valeurs dans `out`. Retourne out_dim, ou -1 en cas d'erreur.
#[no_mangle]
pub unsafe extern "C" fn gopu_model_predict(m: *mut Mlp, x: *const f32, rows: usize, cols: usize, out: *mut f32, out_cap: usize) -> i32 {
    if m.is_null() || x.is_null() || out.is_null() {
        set_err("pointeur nul".to_string());
        return -1;
    }
    let model = &mut *m;
    if cols != model.in_dim() {
        set_err(format!("entree de taille {} attendue, recu {}", model.in_dim(), cols));
        return -1;
    }
    let od = model.out_dim();
    if rows * od > out_cap {
        set_err("tampon de sortie trop petit".to_string());
        return -1;
    }
    let xs = match Tensor::from_vec(std::slice::from_raw_parts(x, rows * cols).to_vec(), &[rows, cols]) {
        Ok(t) => t,
        Err(e) => {
            set_err(e);
            return -1;
        }
    };
    let y = model.forward(&xs);
    std::ptr::copy_nonoverlapping(y.data.as_ptr(), out, rows * od);
    od as i32
}

#[no_mangle]
pub unsafe extern "C" fn gopu_model_save(m: *const Mlp, path: *const c_char) -> i32 {
    if m.is_null() {
        set_err("modele nul".to_string());
        return -1;
    }
    match crate::io::save_model(&*m, &cstr(path)) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_model_load(path: *const c_char) -> *mut Mlp {
    match crate::io::load_model(&cstr(path)) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------- orchestrateur

/// Callback d'agent (Python). Ecrit la reponse UTF-8 dans `out` (capacite `cap`), retourne sa longueur ou -1.
pub type AgentCb = extern "C" fn(
    prompt: *const c_char,
    context: *const c_char,
    deps: *const c_char,
    out: *mut u8,
    cap: usize,
    user: *mut c_void,
) -> i32;

struct CbAgent {
    cb: AgentCb,
    user: usize,
}

impl Agent for CbAgent {
    fn handle(&mut self, input: &TaskInput, _ctx: &Ctx) -> Result<String, String> {
        let p = CString::new(input.prompt.replace('\0', " ")).map_err(|e| e.to_string())?;
        let c = CString::new(input.context.replace('\0', " ")).map_err(|e| e.to_string())?;
        let d = CString::new(input.deps_outputs.join("\n---\n").replace('\0', " ")).map_err(|e| e.to_string())?;
        let mut buf = vec![0u8; 65536];
        let n = (self.cb)(p.as_ptr(), c.as_ptr(), d.as_ptr(), buf.as_mut_ptr(), buf.len(), self.user as *mut c_void);
        if n < 0 {
            return Err("le callback de l'agent a echoue".to_string());
        }
        let n = (n as usize).min(buf.len());
        Ok(String::from_utf8_lossy(&buf[..n]).into_owned())
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_new(session: *const c_char) -> *mut Orchestrator {
    Box::into_raw(Box::new(Orchestrator::new(&cstr(session))))
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_free(o: *mut Orchestrator) {
    if !o.is_null() {
        drop(Box::from_raw(o));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_register_cb(o: *mut Orchestrator, name: *const c_char, cb: AgentCb, user: *mut c_void) -> i32 {
    if o.is_null() {
        return -1;
    }
    (*o).register(&cstr(name), Box::new(CbAgent { cb, user: user as usize }));
    0
}

/// kind : "recall" ou "summarize".
#[no_mangle]
pub unsafe extern "C" fn gopu_orch_register_builtin(o: *mut Orchestrator, name: *const c_char, kind: *const c_char) -> i32 {
    if o.is_null() {
        return -1;
    }
    match cstr(kind).as_str() {
        "recall" => (*o).register(&cstr(name), Box::new(RecallAgent)),
        "summarize" => (*o).register(&cstr(name), Box::new(SummarizeAgent)),
        other => {
            set_err(format!("agent integre inconnu : {}", other));
            return -1;
        }
    }
    0
}

fn parse_tasks(s: &str) -> Result<Vec<TaskSpec>, String> {
    let v = json::parse(s)?;
    let arr = v.as_arr().ok_or("tableau de taches attendu")?;
    let mut out = Vec::new();
    for t in arr {
        let name = t.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let agent = t.get("agent").and_then(|x| x.as_str()).ok_or("champ 'agent' manquant")?.to_string();
        let prompt = t.get("prompt").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let deps: Vec<usize> = match t.get("deps").and_then(|x| x.as_arr()) {
            Some(a) => a.iter().filter_map(|d| d.as_f64()).map(|d| d as usize).collect(),
            None => Vec::new(),
        };
        out.push(TaskSpec { name, agent, prompt, deps });
    }
    Ok(out)
}

/// Execute un tableau JSON de taches. Retourne un JSON de resultats (a liberer avec gopu_string_free) ou NULL.
#[no_mangle]
pub unsafe extern "C" fn gopu_orch_run(o: *mut Orchestrator, tasks_json: *const c_char) -> *mut c_char {
    if o.is_null() {
        set_err("orchestrateur nul".to_string());
        return std::ptr::null_mut();
    }
    let tasks = match parse_tasks(&cstr(tasks_json)) {
        Ok(t) => t,
        Err(e) => {
            set_err(e);
            return std::ptr::null_mut();
        }
    };
    match (*o).run(&tasks) {
        Ok(res) => {
            let arr: Vec<Value> = res
                .iter()
                .map(|r| {
                    Value::Obj(vec![
                        ("id".to_string(), Value::Num(r.id as f64)),
                        ("name".to_string(), Value::Str(r.name.clone())),
                        ("agent".to_string(), Value::Str(r.agent.clone())),
                        ("ok".to_string(), Value::Bool(r.ok)),
                        ("output".to_string(), Value::Str(r.output.clone())),
                        ("attempts".to_string(), Value::Num(r.attempts as f64)),
                    ])
                })
                .collect();
            to_c(json::to_string(&Value::Arr(arr)))
        }
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_cancel(o: *mut Orchestrator) {
    if !o.is_null() {
        (*o).cancel.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_remember(o: *mut Orchestrator, kind: *const c_char, text: *const c_char) -> i64 {
    if o.is_null() {
        return -1;
    }
    let k = match Kind::parse(&cstr(kind)) {
        Some(k) => k,
        None => {
            set_err("type de noeud inconnu".to_string());
            return -1;
        }
    };
    match (*o).graph.lock() {
        Ok(mut g) => g.add_node(k, &cstr(text)) as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_recall(o: *mut Orchestrator, query: *const c_char, k: usize, depth: usize, budget: usize) -> *mut c_char {
    if o.is_null() {
        return std::ptr::null_mut();
    }
    match (*o).graph.lock() {
        Ok(mut g) => to_c(g.recall(&cstr(query), k, depth, budget)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_graph_size(o: *mut Orchestrator) -> i64 {
    if o.is_null() {
        return -1;
    }
    match (*o).graph.lock() {
        Ok(g) => g.len() as i64,
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_graph_save(o: *mut Orchestrator, path: *const c_char) -> i32 {
    if o.is_null() {
        return -1;
    }
    match (*o).graph.lock() {
        Ok(g) => match g.save(&cstr(path)) {
            Ok(()) => 0,
            Err(e) => {
                set_err(e);
                -1
            }
        },
        Err(_) => -1,
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_orch_graph_load(o: *mut Orchestrator, path: *const c_char) -> i32 {
    if o.is_null() {
        return -1;
    }
    match SessionGraph::load(&cstr(path)) {
        Ok(loaded) => match (*o).graph.lock() {
            Ok(mut g) => {
                *g = loaded;
                0
            }
            Err(_) => -1,
        },
        Err(e) => {
            set_err(e);
            -1
        }
    }
}
