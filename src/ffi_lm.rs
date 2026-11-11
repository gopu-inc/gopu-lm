//! Interface C du modele de langage gopu-lm (utilisee par python/gopu_lm via ctypes).
//! Les parametres et resultats complexes passent en JSON ; en cas d'erreur : NULL / -1 + gopu_last_error().

use crate::ffi::{cstr, set_err, to_c};
use crate::json::{self, Value};
use crate::lm::{prepare_data, Config, DataOpts, GenOpts, Lm, TrainOpts};
use crate::tokenizer::Tokenizer;
use std::os::raw::c_char;

fn num(v: &Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| x.as_f64())
}

fn parse_opts(s: &str) -> Result<Value, String> {
    if s.trim().is_empty() {
        return Ok(Value::Obj(vec![]));
    }
    json::parse(s)
}

fn ret(r: Result<String, String>) -> *mut c_char {
    match r {
        Ok(s) => to_c(s),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_lm_load(path: *const c_char) -> *mut Lm {
    match Lm::load(&cstr(path)) {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_lm_free(m: *mut Lm) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gopu_lm_save(m: *const Lm, path: *const c_char) -> i32 {
    if m.is_null() {
        set_err("modele nul".to_string());
        return -1;
    }
    match (*m).save(&cstr(path)) {
        Ok(()) => 0,
        Err(e) => {
            set_err(e);
            -1
        }
    }
}

/// Cree un modele : le tokenizer BPE est appris sur `corpus`. cfg_json : vocab, d, layers, heads, ctx, ff, future, seed.
#[no_mangle]
pub unsafe extern "C" fn gopu_lm_new(cfg_json: *const c_char, corpus: *const c_char) -> *mut Lm {
    let r = (|| -> Result<Lm, String> {
        let v = parse_opts(&cstr(cfg_json))?;
        let def = Config::small(0);
        let g = |k: &str, d: usize| num(&v, k).map(|x| x as usize).unwrap_or(d);
        let d = g("d", def.d);
        let cfg = Config { vocab: 0, d, layers: g("layers", def.layers), heads: g("heads", def.heads), ctx: g("ctx", def.ctx), ff: g("ff", 4 * d), future: g("future", def.future) };
        let tok = Tokenizer::train(&cstr(corpus), g("vocab", 640));
        Lm::new(cfg, tok, num(&v, "seed").unwrap_or(1.0) as u64)
    })();
    match r {
        Ok(m) => Box::into_raw(Box::new(m)),
        Err(e) => {
            set_err(e);
            std::ptr::null_mut()
        }
    }
}

/// Entraine sur un corpus de conversations. Les journaux vont sur stderr. Retourne {"steps","loss","val_loss","seconds"}.
#[no_mangle]
pub unsafe extern "C" fn gopu_lm_train(m: *mut Lm, corpus: *const c_char, opts_json: *const c_char) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    let lm = &mut *m;
    ret((|| {
        let v = parse_opts(&cstr(opts_json))?;
        let d = TrainOpts::default();
        let o = TrainOpts {
            steps: num(&v, "steps").map(|x| x as usize).unwrap_or(d.steps),
            batch: num(&v, "batch").map(|x| x as usize).unwrap_or(d.batch),
            lr: num(&v, "lr").map(|x| x as f32).unwrap_or(d.lr),
            min_lr_frac: num(&v, "min_lr_frac").map(|x| x as f32).unwrap_or(d.min_lr_frac),
            warmup: num(&v, "warmup").map(|x| x as usize).unwrap_or(d.warmup),
            wd: num(&v, "wd").map(|x| x as f32).unwrap_or(d.wd),
            clip: num(&v, "clip").map(|x| x as f32).unwrap_or(d.clip),
            future_weight: num(&v, "future_weight").map(|x| x as f32).unwrap_or(d.future_weight),
            seed: num(&v, "seed").map(|x| x as u64).unwrap_or(d.seed),
            log_every: num(&v, "log_every").map(|x| x as usize).unwrap_or(d.log_every),
            eval_every: num(&v, "eval_every").map(|x| x as usize).unwrap_or(d.eval_every),
            threads: num(&v, "threads").map(|x| x as usize).unwrap_or(d.threads),
            max_seconds: num(&v, "max_seconds").map(|x| x as f32).unwrap_or(d.max_seconds),
        };
        if let Some(c) = num(&v, "ctx") {
            lm.resize_ctx(c as usize)?;
        }
        let dd = DataOpts::default();
        let dopts = DataOpts {
            user_weight: num(&v, "user_weight").map(|x| x as f32).unwrap_or(dd.user_weight),
            holdout_every: num(&v, "holdout_every").map(|x| x as usize).unwrap_or(dd.holdout_every),
            max_corpus_tokens: num(&v, "max_corpus_tokens").map(|x| x as usize).unwrap_or(dd.max_corpus_tokens),
            pack_len: num(&v, "pack_len").map(|x| x as usize).unwrap_or(dd.pack_len),
            seed: o.seed,
        };
        let (tr, va, st) = prepare_data(&lm.tok, &cstr(corpus), &dopts);
        eprintln!("{} conversations -> {} sequences d'entrainement ({} jetons{}), {} de validation", st.conversations, st.train_sequences, st.train_tokens, if st.truncated { ", corpus plafonne" } else { "" }, st.val_sequences);
        let rep = lm.train(&tr, &va, &o, &mut |s| eprintln!("{}", s))?;
        Ok(json::to_string(&Value::Obj(vec![
            ("steps".into(), Value::Num(rep.steps as f64)),
            ("loss".into(), Value::Num(rep.loss as f64)),
            ("val_loss".into(), rep.val_loss.map(|x| Value::Num(x as f64)).unwrap_or(Value::Null)),
            ("seconds".into(), Value::Num(rep.seconds as f64)),
            ("train_tokens".into(), Value::Num(st.train_tokens as f64)),
            ("train_sequences".into(), Value::Num(st.train_sequences as f64)),
            ("conversations".into(), Value::Num(st.conversations as f64)),
            ("corpus_truncated".into(), Value::Bool(st.truncated)),
        ])))
    })())
}

/// Genere la suite d'un prompt deja formate. Retourne {"text","finish","stop","n_tokens"}.
#[no_mangle]
pub unsafe extern "C" fn gopu_lm_generate(m: *const Lm, prompt: *const c_char, opts_json: *const c_char) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    let lm = &*m;
    ret((|| {
        let v = parse_opts(&cstr(opts_json))?;
        let mut o = GenOpts::default();
        if let Some(x) = num(&v, "max_tokens") {
            o.max_tokens = x as usize;
        }
        if let Some(x) = num(&v, "temperature") {
            o.temperature = x as f32;
        }
        if let Some(x) = num(&v, "top_k") {
            o.top_k = x as usize;
        }
        if let Some(x) = num(&v, "top_p") {
            o.top_p = x as f32;
        }
        if let Some(x) = num(&v, "repetition_penalty") {
            o.repetition_penalty = x as f32;
        }
        if let Some(x) = num(&v, "seed") {
            o.seed = x as u64;
        }
        if let Some(a) = v.get("stop").and_then(|s| s.as_arr()) {
            o.stop_strings = a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect();
        }
        let out = lm.generate(&lm.tok.encode(&cstr(prompt)), &o);
        Ok(json::to_string(&Value::Obj(vec![
            ("text".into(), Value::Str(lm.tok.decode(&out.tokens))),
            ("finish".into(), Value::Str(out.finish.to_string())),
            ("stop".into(), out.stop.map(Value::Str).unwrap_or(Value::Null)),
            ("n_tokens".into(), Value::Num(out.tokens.len() as f64)),
        ])))
    })())
}

/// Predictions des tetes "futur" : liste (un element par horizon) de [{"token","id","p"}].
#[no_mangle]
pub unsafe extern "C" fn gopu_lm_future(m: *const Lm, prompt: *const c_char, top_n: usize) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    let lm = &*m;
    let ids = lm.tok.encode(&cstr(prompt));
    let heads: Vec<Value> = lm
        .predict_future(&ids, top_n)
        .into_iter()
        .map(|top| {
            Value::Arr(
                top.into_iter()
                    .map(|(id, p)| Value::Obj(vec![("token".into(), Value::Str(lm.tok.decode(&[id]))), ("id".into(), Value::Num(id as f64)), ("p".into(), Value::Num(p as f64))]))
                    .collect(),
            )
        })
        .collect();
    to_c(json::to_string(&Value::Arr(heads)))
}

#[no_mangle]
pub unsafe extern "C" fn gopu_lm_encode(m: *const Lm, text: *const c_char) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    let ids = (*m).tok.encode(&cstr(text));
    to_c(json::to_string(&Value::Arr(ids.into_iter().map(|i| Value::Num(i as f64)).collect())))
}

#[no_mangle]
pub unsafe extern "C" fn gopu_lm_decode(m: *const Lm, ids_json: *const c_char) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    ret((|| {
        let v = json::parse(&cstr(ids_json))?;
        let ids: Vec<u32> = v.as_arr().ok_or("liste d'ids attendue")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as u32).collect();
        Ok((*m).tok.decode(&ids))
    })())
}

/// mode = "tokens" (embeddings statiques) ou "hidden" (etats caches contextuels). Retourne {"shape":[n,d],"data":[...]}.
#[no_mangle]
pub unsafe extern "C" fn gopu_lm_embed(m: *const Lm, text: *const c_char, mode: *const c_char) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    let lm = &*m;
    let ids = lm.tok.encode(&cstr(text));
    let t = if cstr(mode) == "hidden" { lm.hidden_states(&ids) } else { lm.token_embeddings(&ids) };
    to_c(json::to_string(&Value::Obj(vec![
        ("shape".into(), Value::Arr(t.shape.iter().map(|&s| Value::Num(s as f64)).collect())),
        ("data".into(), Value::Arr(t.data.iter().map(|&x| Value::Num(x as f64)).collect())),
    ])))
}

#[no_mangle]
pub unsafe extern "C" fn gopu_lm_info(m: *const Lm) -> *mut c_char {
    if m.is_null() {
        set_err("modele nul".to_string());
        return std::ptr::null_mut();
    }
    let c = (*m).cfg();
    let f = |k: &str, v: usize| (k.to_string(), Value::Num(v as f64));
    to_c(json::to_string(&Value::Obj(vec![f("vocab", c.vocab), f("d", c.d), f("layers", c.layers), f("heads", c.heads), f("ctx", c.ctx), f("ff", c.ff), f("future", c.future), f("n_params", c.n_params())])))
}
