//! CLI du modele de langage gopu-lm : entrainer, interroger, inspecter un fichier .b8n / .safetensors.
//!
//!   gopu-lm train --data data/train.txt --out modele.b8n [--resume modele.b8n] [--steps 3000] ...
//!   gopu-lm ask   --model modele.b8n --prompt "Bonjour" [--temperature 0] [--max-tokens 160]
//!   gopu-lm info  --model modele.b8n

use gopu::lm::{prepare_data, Config, DataOpts, GenOpts, Lm, TrainOpts};
use gopu::tokenizer::{self, Tokenizer};
use std::collections::HashMap;
use std::fs;

fn parse_args(args: &[String]) -> Result<HashMap<String, String>, String> {
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

fn get<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str, default: T) -> Result<T, String> {
    match m.get(k) {
        Some(v) => v.parse::<T>().map_err(|_| format!("--{} : valeur invalide '{}'", k, v)),
        None => Ok(default),
    }
}

fn need<'a>(m: &'a HashMap<String, String>, k: &str) -> Result<&'a String, String> {
    m.get(k).ok_or(format!("--{} est obligatoire", k))
}

fn cmd_train(m: &HashMap<String, String>) -> Result<(), String> {
    let data = need(m, "data")?;
    let out = need(m, "out")?;
    let text = fs::read_to_string(data).map_err(|e| format!("{} : {}", data, e))?;
    let mut lm = match m.get("resume") {
        Some(p) => {
            let lm = Lm::load(p)?;
            eprintln!("reprise depuis {} ({} parametres, vocabulaire {})", p, lm.cfg().n_params(), lm.cfg().vocab);
            lm
        }
        None => {
            let vocab: usize = get(m, "vocab", 640)?;
            let tok = Tokenizer::train(&text, vocab);
            let def = Config::small(0);
            let cfg = Config {
                vocab: 0,
                d: get(m, "d", def.d)?,
                layers: get(m, "layers", def.layers)?,
                heads: get(m, "heads", def.heads)?,
                ctx: get(m, "ctx", def.ctx)?,
                ff: get(m, "ff", 4 * get::<usize>(m, "d", def.d)?)?,
                future: get(m, "future", def.future)?,
            };
            let lm = Lm::new(cfg, tok, get(m, "seed", 1u64)?)?;
            eprintln!("nouveau modele : {} parametres, vocabulaire {}, contexte {}", lm.cfg().n_params(), lm.cfg().vocab, lm.cfg().ctx);
            lm
        }
    };
    if m.contains_key("resume") {
        if let Some(c) = m.get("ctx") {
            lm.resize_ctx(c.parse::<usize>().map_err(|_| "--ctx invalide".to_string())?)?;
        }
    }
    let dd = DataOpts::default();
    let dopts = DataOpts {
        user_weight: get(m, "user-weight", dd.user_weight)?,
        holdout_every: get(m, "holdout-every", dd.holdout_every)?,
        max_corpus_tokens: get(m, "max-corpus-tokens", dd.max_corpus_tokens)?,
        pack_len: get(m, "pack-len", dd.pack_len)?,
        seed: get(m, "seed", 1u64)?,
    };
    let (samples, val, st) = prepare_data(&lm.tok, &text, &dopts);
    eprintln!("{} conversations -> {} sequences d'entrainement ({} jetons{}), {} de validation", st.conversations, st.train_sequences, st.train_tokens, if st.truncated { ", corpus plafonne" } else { "" }, st.val_sequences);
    let d = TrainOpts::default();
    let o = TrainOpts {
        steps: get(m, "steps", d.steps)?,
        batch: get(m, "batch", d.batch)?,
        lr: get(m, "lr", d.lr)?,
        min_lr_frac: get(m, "min-lr-frac", d.min_lr_frac)?,
        warmup: get(m, "warmup", d.warmup)?,
        wd: get(m, "wd", d.wd)?,
        clip: get(m, "clip", d.clip)?,
        future_weight: get(m, "future-weight", d.future_weight)?,
        seed: get(m, "seed", d.seed)?,
        log_every: get(m, "log-every", d.log_every)?,
        eval_every: get(m, "eval-every", d.eval_every)?,
        threads: get(m, "threads", d.threads)?,
        max_seconds: get(m, "max-seconds", d.max_seconds)?,
    };
    let rep = lm.train(&samples, &val, &o, &mut |s| eprintln!("{}", s))?;
    lm.save(out)?;
    eprintln!("termine : {} etapes en {:.0}s | perte {:.3} | validation {} -> {}", rep.steps, rep.seconds, rep.loss, rep.val_loss.map(|v| format!("{:.3}", v)).unwrap_or("-".into()), out);
    Ok(())
}

fn cmd_ask(m: &HashMap<String, String>) -> Result<(), String> {
    let lm = Lm::load(need(m, "model")?)?;
    let prompt = need(m, "prompt")?;
    let mut ctx = String::new();
    if let Some(sys) = m.get("system") {
        ctx.push_str(&format!("<|system|>\n{}\n</|system|>\n", sys));
    }
    ctx.push_str(&format!("<|user|>\n{}\n</|user|>\n", prompt));
    let mut o = GenOpts::default();
    o.max_tokens = get(m, "max-tokens", o.max_tokens)?;
    o.temperature = get(m, "temperature", o.temperature)?;
    o.top_k = get(m, "top-k", o.top_k)?;
    o.top_p = get(m, "top-p", o.top_p)?;
    o.repetition_penalty = get(m, "repetition-penalty", o.repetition_penalty)?;
    o.seed = get(m, "seed", o.seed)?;
    let out = lm.generate(&lm.tok.encode(&ctx), &o);
    println!("{}", lm.tok.decode(&out.tokens));
    eprintln!("[{} jetons, arret : {}]", out.tokens.len(), out.finish);
    if m.get("future").map(|v| v == "1").unwrap_or(false) {
        let ids = lm.tok.encode(&ctx);
        for (h, top) in lm.predict_future(&ids, 3).iter().enumerate() {
            let s: Vec<String> = top.iter().map(|(id, p)| format!("{:?} {:.2}", lm.tok.decode(&[*id]), p)).collect();
            eprintln!("futur +{} : {}", h + 1, s.join(" | "));
        }
    }
    Ok(())
}

fn cmd_info(m: &HashMap<String, String>) -> Result<(), String> {
    let path = need(m, "model")?;
    let lm = Lm::load(path)?;
    let c = lm.cfg();
    println!("fichier       : {}", path);
    println!("architecture  : gopu-lm (Transformer pre-norm, tetes partagees avec les embeddings)");
    println!("parametres    : {}", c.n_params());
    println!("vocabulaire   : {} ({} octets + {} speciaux + {} fusions BPE)", c.vocab, 256, tokenizer::SPECIALS.len(), c.vocab - tokenizer::base_vocab() as usize);
    println!("d / couches   : {} / {} ({} tetes, ff {})", c.d, c.layers, c.heads, c.ff);
    println!("contexte      : {} jetons", c.ctx);
    println!("tetes futur   : {} (predisent les jetons t+2 .. t+{})", c.future, c.future + 1);
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let res = (|| -> Result<(), String> {
        let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("");
        let m = parse_args(&args[args.len().min(2)..])?;
        match cmd {
            "train" => cmd_train(&m),
            "ask" => cmd_ask(&m),
            "info" => cmd_info(&m),
            _ => Err("usage : gopu-lm <train|ask|info> --option valeur ...".to_string()),
        }
    })();
    if let Err(e) = res {
        eprintln!("erreur : {}", e);
        std::process::exit(1);
    }
}
