//! CLI du modele airtel (Llama, Rust pur) : init, train, lora, merge, sample, dpo, ask, info.
//!
//!   airtel init   --data data/train.txt --out airtel-2.1m [--vocab 4096] [--ctx 20000]
//!   airtel train  --model airtel-2.1m --data data/train.txt [--steps N] [--out dossier]
//!   airtel lora   --model airtel-2.1m --data polite.txt --adapter-out adapters/poli [--r 16] [--merge-out dossier]
//!   airtel merge  --model airtel-2.1m --adapter adapters/poli --out airtel-2.1m-poli
//!   airtel sample --model dossier [--adapter a] --prompts prompts.txt --n 4 --out samples.jsonl
//!   airtel dpo    --model dossier --pairs prefs.jsonl --adapter-out adapters/dpo [--merge-out dossier]
//!   airtel ask    --model dossier [--adapter a] --prompt "Bonjour"
//!   airtel info   --model dossier

use gopu::airtel::{load_pairs, AConfig, Airtel, DpoOpts, Lora, LoraCfg, Mode};
use gopu::lm::{prepare_data, DataOpts, GenOpts, TrainOpts};
use gopu::tokenizer::{self, PreTok, Tokenizer};
use std::collections::HashMap;
use std::fs;

type M = HashMap<String, String>;

fn parse_args(args: &[String]) -> Result<M, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let k = args[i].strip_prefix("--").ok_or(format!("argument inattendu : {}", args[i]))?;
        let v = args.get(i + 1).ok_or(format!("valeur manquante pour --{}", k))?;
        m.insert(k.to_string(), v.clone());
        i += 2;
    }
    Ok(m)
}
fn get<T: std::str::FromStr>(m: &M, k: &str, d: T) -> Result<T, String> {
    match m.get(k) {
        Some(v) => v.parse::<T>().map_err(|_| format!("--{} : valeur invalide '{}'", k, v)),
        None => Ok(d),
    }
}
fn need<'a>(m: &'a M, k: &str) -> Result<&'a String, String> {
    m.get(k).ok_or(format!("--{} est obligatoire", k))
}
fn log(s: &str) {
    eprintln!("{}", s);
}

fn lora_cfg(m: &M) -> Result<LoraCfg, String> {
    let d = LoraCfg::default();
    let targets = match m.get("targets") {
        Some(t) if t == "all" => gopu::airtel::LORA_KINDS.iter().map(|s| s.to_string()).collect(),
        Some(t) => t.split(',').map(|s| s.trim().to_string()).collect(),
        None => d.targets.clone(),
    };
    Ok(LoraCfg { r: get(m, "r", d.r)?, alpha: get(m, "alpha", d.alpha)?, dropout: get(m, "dropout", d.dropout)?, targets })
}

fn train_opts(m: &M) -> Result<TrainOpts, String> {
    let d = TrainOpts::default();
    Ok(TrainOpts {
        steps: get(m, "steps", d.steps)?,
        batch: get(m, "batch", d.batch)?,
        lr: get(m, "lr", d.lr)?,
        min_lr_frac: get(m, "min-lr-frac", d.min_lr_frac)?,
        warmup: get(m, "warmup", d.warmup)?,
        wd: get(m, "wd", d.wd)?,
        clip: get(m, "clip", d.clip)?,
        future_weight: 0.0,
        seed: get(m, "seed", d.seed)?,
        log_every: get(m, "log-every", d.log_every)?,
        eval_every: get(m, "eval-every", d.eval_every)?,
        threads: get(m, "threads", d.threads)?,
        max_seconds: get(m, "max-seconds", d.max_seconds)?,
    })
}

fn data_opts(m: &M) -> Result<DataOpts, String> {
    let d = DataOpts::default();
    Ok(DataOpts {
        user_weight: get(m, "user-weight", d.user_weight)?,
        holdout_every: get(m, "holdout-every", d.holdout_every)?,
        max_corpus_tokens: get(m, "max-corpus-tokens", 1_000_000)?,
        pack_len: get(m, "pack-len", 512)?,
        seed: get(m, "seed", 1u64)?,
    })
}

fn load_model(m: &M) -> Result<Airtel, String> {
    let mut a = Airtel::load(need(m, "model")?)?;
    if let Some(ad) = m.get("adapter") {
        a.lora = Some(Lora::load(&a.cfg, ad)?);
    }
    Ok(a)
}

fn finish(a: &mut Airtel, m: &M, base: &str) -> Result<(), String> {
    if let Some(dir) = m.get("adapter-out") {
        if let Some(lo) = &a.lora {
            lo.save(dir, base)?;
            log(&format!("adaptateur LoRA ecrit : {} ({} parametres)", dir, lo.n_params()));
        }
    }
    let out = m.get("merge-out").or_else(|| m.get("out"));
    if let Some(dir) = out {
        if a.lora.is_some() {
            a.merge_lora()?;
        }
        a.export_dir(dir, dir)?;
        log(&format!("modele ecrit : {}/ (model.safetensors, model.b8n, config, tokenizer)", dir));
    }
    Ok(())
}

fn cmd_init(m: &M) -> Result<(), String> {
    let text = fs::read_to_string(need(m, "data")?).map_err(|e| e.to_string())?;
    let out = need(m, "out")?;
    let vocab: usize = get(m, "vocab", 4096)?;
    log(&format!("entrainement du tokenizer BPE (vocabulaire {}) ...", vocab));
    let tok = Tokenizer::train_with(&text, vocab, PreTok::Gpt);
    let mut cfg = AConfig::airtel_2_1m(vocab);
    cfg.ctx = get(m, "ctx", cfg.ctx)?;
    cfg.d = get(m, "d", cfg.d)?;
    cfg.layers = get(m, "layers", cfg.layers)?;
    cfg.heads = get(m, "heads", cfg.heads)?;
    cfg.ff = get(m, "ff", cfg.ff)?;
    let mut a = Airtel::new(cfg, tok, get(m, "seed", 1u64)?)?;
    log(&format!("modele : {} parametres, vocabulaire {} (lignes {}), contexte {}", a.n_params(), a.tok.vocab_size(), a.cfg.vocab, a.cfg.ctx));
    finish(&mut a, m, "airtel")
}

fn cmd_train(m: &M, lora: bool) -> Result<(), String> {
    let mut a = load_model(m)?;
    let text = fs::read_to_string(need(m, "data")?).map_err(|e| e.to_string())?;
    let (tr, va, st) = prepare_data(&a.tok, &text, &data_opts(m)?);
    log(&format!("{} conversations -> {} sequences ({} jetons{}), {} de validation", st.conversations, st.train_sequences, st.train_tokens, if st.truncated { ", plafonne" } else { "" }, st.val_sequences));
    let o = train_opts(m)?;
    let mode = if lora {
        if a.lora.is_none() {
            a.add_lora(lora_cfg(m)?, get(m, "seed", 1u64)?)?;
        }
        log(&format!("LoRA : {} parametres entrainables sur {}", a.lora.as_ref().unwrap().n_params(), a.n_params()));
        Mode::Lora
    } else {
        Mode::Full
    };
    let r = a.train(mode, &tr, &va, &o, &mut log)?;
    log(&format!("termine : {} etapes en {:.0}s | perte {:.3} | validation {}", r.steps, r.seconds, r.loss, r.val_loss.map(|v| format!("{:.3}", v)).unwrap_or("-".into())));
    finish(&mut a, m, m.get("model").map(|s| s.as_str()).unwrap_or("airtel"))
}

fn cmd_merge(m: &M) -> Result<(), String> {
    let mut a = load_model(m)?;
    if a.lora.is_none() {
        return Err("--adapter est obligatoire".to_string());
    }
    finish(&mut a, m, "airtel")
}

fn chat_prompt(sys: Option<&String>, user: &str) -> String {
    let mut s = String::new();
    if let Some(x) = sys {
        s.push_str(&format!("<|system|>\n{}\n</|system|>\n", x));
    }
    s.push_str(&format!("<|user|>\n{}\n</|user|>\n", user));
    s
}

fn gen_opts(m: &M) -> Result<GenOpts, String> {
    let mut o = GenOpts::default();
    o.max_tokens = get(m, "max-tokens", 160)?;
    o.temperature = get(m, "temperature", o.temperature)?;
    o.top_k = get(m, "top-k", o.top_k)?;
    o.top_p = get(m, "top-p", o.top_p)?;
    o.repetition_penalty = get(m, "repetition-penalty", o.repetition_penalty)?;
    o.seed = get(m, "seed", o.seed)?;
    o.stop_ids = vec![tokenizer::EOS, tokenizer::ASSIST_CLOSE];
    Ok(o)
}

fn cmd_ask(m: &M) -> Result<(), String> {
    let a = load_model(m)?;
    let p = chat_prompt(m.get("system"), need(m, "prompt")?);
    let out = a.generate(&a.tok.encode(&p), &gen_opts(m)?);
    println!("{}", a.tok.decode(&out.tokens).replace("<|eos|>", "").trim());
    eprintln!("[{} jetons, arret : {}]", out.tokens.len(), out.finish);
    Ok(())
}

/// Genere N reponses par prompt (une ligne = un prompt) -> JSONL {"prompt","answers":[...]}.
fn cmd_sample(m: &M) -> Result<(), String> {
    let a = load_model(m)?;
    let prompts = fs::read_to_string(need(m, "prompts")?).map_err(|e| e.to_string())?;
    let n: usize = get(m, "n", 4)?;
    let mut o = gen_opts(m)?;
    o.temperature = get(m, "temperature", 0.9)?;
    let base_seed: u64 = get(m, "seed", 1u64)?;
    let mut out = String::new();
    for (pi, line) in prompts.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let enc = a.tok.encode(&chat_prompt(m.get("system"), line.trim()));
        let mut answers = Vec::new();
        for k in 0..n {
            o.seed = base_seed * 1000 + (pi * n + k) as u64 + 1;
            let g = a.generate(&enc, &o);
            let t = a.tok.decode(&g.tokens);
            answers.push(gopu::json::Value::Str(t.replace("<|eos|>", "").trim().to_string()));
        }
        let v = gopu::json::Value::Obj(vec![("prompt".into(), gopu::json::Value::Str(line.trim().to_string())), ("answers".into(), gopu::json::Value::Arr(answers))]);
        out.push_str(&gopu::json::to_string(&v));
        out.push('\n');
    }
    fs::write(need(m, "out")?, out).map_err(|e| e.to_string())
}

fn cmd_dpo(m: &M) -> Result<(), String> {
    let mut a = load_model(m)?;
    let pairs = load_pairs(&a.tok, &fs::read_to_string(need(m, "pairs")?).map_err(|e| e.to_string())?)?;
    log(&format!("{} paires de preferences", pairs.len()));
    let d = DpoOpts::default();
    let o = DpoOpts {
        beta: get(m, "beta", d.beta)?,
        epochs: get(m, "epochs", d.epochs)?,
        batch: get(m, "batch", d.batch)?,
        lr: get(m, "lr", d.lr)?,
        sft_alpha: get(m, "sft-alpha", d.sft_alpha)?,
        lora: lora_cfg(m)?,
        seed: get(m, "seed", 1u64)?,
        log_every: get(m, "log-every", d.log_every)?,
        threads: get(m, "threads", 0)?,
    };
    let r = a.dpo(&pairs, &o, &mut log)?;
    log(&format!("DPO termine : {} etapes en {:.0}s | precision {:.0}% | marge moyenne {:+.2}", r.steps, r.seconds, r.accuracy * 100.0, r.margin));
    finish(&mut a, m, m.get("model").map(|s| s.as_str()).unwrap_or("airtel"))
}

/// Logits du jeton suivant (JSON) pour une suite d'identifiants : sert aux verifications croisees.
fn cmd_logits(m: &M) -> Result<(), String> {
    let a = load_model(m)?;
    let ids: Vec<u32> = if let Some(t) = m.get("text") { a.tok.encode(t) } else { need(m, "ids")?.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse::<u32>().map_err(|e| e.to_string())).collect::<Result<_, _>>()? };
    let lg = a.last_logits(&ids);
    let ids_s: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    let lg_s: Vec<String> = lg.iter().map(|x| format!("{:.6}", x)).collect();
    println!("{{\"ids\":[{}],\"logits\":[{}]}}", ids_s.join(","), lg_s.join(","));
    Ok(())
}

fn cmd_info(m: &M) -> Result<(), String> {
    let a = load_model(m)?;
    let c = &a.cfg;
    println!("architecture : airtel (Llama : RMSNorm, RoPE, SwiGLU, sans biais, embeddings lies)");
    println!("parametres   : {}", a.n_params());
    println!("vocabulaire  : {} utilises / {} lignes", a.tok.vocab_size(), c.vocab);
    println!("d / couches  : {} / {} ({} tetes, ff {})", c.d, c.layers, c.heads, c.ff);
    println!("contexte     : {} jetons (RoPE theta {})", c.ctx, c.rope_theta);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let res = (|| -> Result<(), String> {
        let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
        let m = parse_args(&args[args.len().min(2)..])?;
        match cmd {
            "init" => cmd_init(&m),
            "train" => cmd_train(&m, false),
            "lora" => cmd_train(&m, true),
            "merge" => cmd_merge(&m),
            "sample" => cmd_sample(&m),
            "dpo" => cmd_dpo(&m),
            "ask" => cmd_ask(&m),
            "info" => cmd_info(&m),
            "logits" => cmd_logits(&m),
            _ => Err("usage : airtel <init|train|lora|merge|sample|dpo|ask|info|logits> --option valeur ...".to_string()),
        }
    })();
    if let Err(e) = res {
        eprintln!("erreur : {}", e);
        std::process::exit(1);
    }
}
