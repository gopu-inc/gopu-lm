use gopu::graph::Kind;
use gopu::io::{load_model, save_model};
use gopu::nn::{Act, Mlp};
use gopu::orch::{Ctx, FnAgent, Orchestrator, RecallAgent, SummarizeAgent, TaskInput, TaskSpec};
use gopu::tensor::Tensor;

fn main() -> Result<(), String> {
    println!("== 1. Entrainement leger (XOR) ==");
    let x = Tensor::from_vec(vec![0., 0., 0., 1., 1., 0., 1., 1.], &[4, 2])?;
    let y = [0usize, 1, 1, 0];
    let mut model = Mlp::new(&[2, 8, 2], &[Act::Tanh, Act::None], 42)?;
    let hist = model.fit(&x, &y, 1500, 0.05, 4)?;
    println!("perte : {:.4} -> {:.4}", hist[0], hist[hist.len() - 1]);

    println!("== 2. save_model (.safetensors et .b8n) ==");
    let dir = std::env::temp_dir();
    let st = dir.join("gopu_demo.safetensors");
    let b8 = dir.join("gopu_demo.b8n");
    save_model(&model, st.to_str().ok_or("chemin invalide")?)?;
    save_model(&model, b8.to_str().ok_or("chemin invalide")?)?;
    let mut m2 = load_model(b8.to_str().ok_or("chemin invalide")?)?;
    let out = m2.forward(&x);
    let mut correct = 0;
    for i in 0..4 {
        let (a, b) = (out.data[i * 2], out.data[i * 2 + 1]);
        let pred = if b > a { 1 } else { 0 };
        if pred == y[i] {
            correct += 1;
        }
    }
    println!("modele recharge depuis .b8n : {}/4 corrects", correct);

    println!("== 3. Orchestrateur + graphe de session ==");
    let mut orch = Orchestrator::new("demo");
    {
        let mut g = orch.graph.lock().map_err(|_| "verrou".to_string())?;
        g.add_node(Kind::Fact, "Le framework gopu.inc est ecrit en Rust sans dependance lourde.");
        g.add_node(Kind::Fact, "Une session utilise un seul graphe qui garde les connaissances des agents.");
    }
    orch.register("recherche", Box::new(RecallAgent));
    orch.register("synthese", Box::new(SummarizeAgent));
    orch.register(
        "majuscules",
        Box::new(FnAgent::new(|i: &TaskInput, _c: &Ctx| Ok(i.deps_outputs.join(" ").to_uppercase()))),
    );
    let tasks = vec![
        TaskSpec { name: "chercher".into(), agent: "recherche".into(), prompt: "graphe de session gopu".into(), deps: vec![] },
        TaskSpec { name: "resumer".into(), agent: "synthese".into(), prompt: "resume".into(), deps: vec![0] },
        TaskSpec { name: "formater".into(), agent: "majuscules".into(), prompt: "".into(), deps: vec![1] },
    ];
    for r in orch.run(&tasks)? {
        println!("[{}] {} (agent {}, essais {}) -> {}", r.id, r.name, r.agent, r.attempts, r.output.replace('\n', " / "));
    }
    let g = orch.graph.lock().map_err(|_| "verrou".to_string())?;
    println!("graphe : {} noeuds, {} aretes", g.len(), g.edge_count());
    Ok(())
}
