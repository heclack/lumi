/// Lumi Mamba-3 native GPU inference (HIP/ROCm). Direct kernel calls onto
/// `GpuBuf`s — no framework — following the architecture decision recorded in
/// `inference/METAL_TO_HIP.md`.
///
/// Subcommands:
///   generate   Generate text from a prompt, streaming tokens to stdout.
///   smoke      Random-weight smoke test; needs no checkpoint files.
mod config;
mod model;
mod sampler;
mod state;
mod weights;

use clap::{Args, Parser, Subcommand};
use lumi_kernels::ops;

use config::ModelConfig;
use model::Model;
use sampler::Sampler;

#[derive(Parser)]
#[command(name = "lumi-infer", about = "Lumi Mamba-3 native GPU inference (HIP)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text from a prompt.
    Generate(GenerateArgs),
    /// Random-weight smoke test — no checkpoint required. Runs 16 decode steps
    /// from a fixed bos token and checks every logits vector is finite.
    Smoke {
        /// Optional config JSON; defaults to a small byte-level config.
        #[arg(long)]
        config: Option<String>,
    },
}

#[derive(Args)]
struct GenerateArgs {
    #[arg(long)]
    weights: String,
    #[arg(long)]
    config: String,
    /// Tokenizer JSON path (required unless the config declares byte_level).
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Generate(args) => run_generate(&args),
        Command::Smoke { config } => run_smoke(config.as_deref()),
    }
}

fn run_generate(args: &GenerateArgs) -> anyhow::Result<()> {
    eprintln!("Loading config: {}", args.config);
    let model_config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(&args.config)?)?;
    model_config.validate()?;
    eprintln!(
        "Model: d_model={}, layers={}, heads={}, vocab={}{}",
        model_config.d_model,
        model_config.n_layers,
        model_config.n_heads,
        model_config.vocab_size,
        if model_config.byte_level { " (byte-level)" } else { "" }
    );

    eprintln!("Loading weights: {}", args.weights);
    let model_weights = weights::ModelWeights::load(&args.weights, &model_config)?;
    eprintln!("Weights loaded.");

    // Model::new resets all SSM state; a future multi-prompt loop would call
    // model.reset_state() between prompts.
    let mut model = Model::new(model_config, model_weights);

    // Encode the prompt and pick an eos id, based on the config's tokenization mode.
    let byte_level = model.config().byte_level;
    let (prompt_ids, eos_id, tok): (Vec<u32>, u32, Option<tokenizers::Tokenizer>) =
        if byte_level {
            let ids: Vec<u32> = args.prompt.as_bytes().iter().map(|&b| b as u32 + 3).collect();
            (ids, 2u32, None)
        } else {
            let tok_path = args.tokenizer.as_deref()
                .ok_or_else(|| anyhow::anyhow!("--tokenizer is required (config is not byte_level)"))?;
            let tok = tokenizers::Tokenizer::from_file(tok_path)
                .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
            let encoding = tok
                .encode(args.prompt.as_str(), false)
                .map_err(|e| anyhow::anyhow!("failed to encode prompt: {e}"))?;
            let ids = encoding.get_ids().to_vec();
            let eos = tok.token_to_id("<eos>").unwrap_or(2);
            (ids, eos, Some(tok))
        };
    eprintln!(
        "Prompt: {} tokens{}",
        prompt_ids.len(),
        if byte_level { " (byte-level)" } else { "" }
    );
    if prompt_ids.len() > model.config().max_seq_len {
        eprintln!(
            "warning: prompt ({} tokens) exceeds max_seq_len ({}) the model was trained for \
             — persistent SSM state has no hard length limit, but quality may degrade",
            prompt_ids.len(),
            model.config().max_seq_len
        );
    }

    let mut sampler = Sampler::new(args.temperature, args.top_k, args.seed);

    unsafe {
        ops::cublas_init();
    }

    // Prefill: run every prompt token through the model, keeping only the
    // logits from the last one — each call still advances SSM state for every
    // token in between, which is the only thing prefill needs.
    let start = std::time::Instant::now();
    let mut last_logits: Option<Vec<f32>> = None;
    for &token_id in &prompt_ids {
        last_logits = Some(model.forward_step(token_id));
    }
    let prompt_time = start.elapsed();
    eprintln!(
        "Prompt processed in {:.1}ms ({:.0} tok/s)",
        prompt_time.as_millis(),
        prompt_ids.len() as f64 / prompt_time.as_secs_f64().max(1e-9)
    );

    print!("{}", args.prompt);
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut generated = 0usize;
    let gen_start = std::time::Instant::now();
    for _ in 0..args.max_tokens {
        let logits = match last_logits.take() {
            Some(l) => l,
            None => break, // empty prompt — nothing to seed generation with
        };
        let next_token = sampler.sample(&logits);
        if next_token == eos_id {
            break;
        }
        generated += 1;

        if byte_level {
            if (3..259).contains(&next_token) {
                let byte = (next_token - 3) as u8;
                if let Ok(ch) = std::str::from_utf8(&[byte]) {
                    print!("{ch}");
                }
            }
        } else if let Some(ref tok) = tok {
            if let Ok(text) = tok.decode(&[next_token], true) {
                print!("{text}");
            }
        }
        std::io::stdout().flush()?;

        last_logits = Some(model.forward_step(next_token));
    }
    let gen_time = gen_start.elapsed();
    println!();
    eprintln!(
        "Generated {} tokens in {:.1}ms ({:.1} tok/s)",
        generated,
        gen_time.as_millis(),
        generated as f64 / gen_time.as_secs_f64().max(1e-9)
    );

    unsafe {
        ops::cublas_destroy();
    }
    Ok(())
}

fn run_smoke(config_path: Option<&str>) -> anyhow::Result<()> {
    let model_config = match config_path {
        Some(path) => {
            eprintln!("Loading config: {path}");
            let cfg: ModelConfig = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            cfg
        }
        None => {
            eprintln!("No config given — using smoke default");
            ModelConfig::smoke_default()
        }
    };
    model_config.validate()?;
    // Printed from the constructed config, so this can't go stale if
    // smoke_default's values change.
    eprintln!(
        "Model: d_model={}, layers={}, heads={}, d_state={}, vocab={}",
        model_config.d_model, model_config.n_layers, model_config.n_heads,
        model_config.d_state, model_config.vocab_size
    );

    eprintln!("Building random weights (seed=42)...");
    let model_weights = weights::ModelWeights::random(&model_config, 42);
    let mut model = Model::new(model_config, model_weights);

    unsafe {
        ops::cublas_init();
    }

    // Autoregressive decode from bos=1, feeding each argmax token back in.
    // temperature 0 makes Sampler greedy — same argmax path generate uses.
    let mut sampler = Sampler::new(0.0, None, 0);
    let mut token_id = 1u32; // bos
    let mut argmax_ids = Vec::with_capacity(16);
    for step in 0..16 {
        let logits = model.forward_step(token_id);
        let all_finite = logits.iter().all(|v| v.is_finite());
        anyhow::ensure!(all_finite, "step {step}: logits contain non-finite values");

        let next = sampler.sample(&logits);
        argmax_ids.push(next);
        token_id = next;
    }

    unsafe {
        ops::cublas_destroy();
    }

    println!("smoke OK");
    println!("argmax token ids: {argmax_ids:?}");
    Ok(())
}
