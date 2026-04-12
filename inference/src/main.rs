/// Lumi Inference — hybrid Mamba-3 + Attention on Apple Silicon.
///
/// Subcommands:
///   generate   Generate text from a prompt
///   evaluate   Run evaluation benchmarks (perplexity, MC accuracy)

mod model;
mod eval;
mod metal_ssm;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::{Parser, Subcommand};
use model::{ModelConfig, NmModel};

#[derive(Parser)]
#[command(name = "lumi-infer", about = "Lumi inference + evaluation")]
struct Cli {
    /// Use CPU instead of Metal GPU
    #[arg(long, global = true)]
    cpu: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate text from a prompt.
    Generate {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        config: String,
        /// Tokenizer JSON path (required for BPE mode, omit for byte-level).
        #[arg(short, long)]
        tokenizer: Option<String>,
        #[arg(short, long)]
        prompt: String,
        #[arg(long, default_value = "200")]
        max_tokens: usize,
        #[arg(long, default_value = "0.8")]
        temperature: f64,
    },
    /// Run evaluation benchmarks.
    Evaluate {
        #[arg(short, long)]
        model: String,
        #[arg(short, long)]
        config: String,
        /// Tokenizer JSON path (required for BPE mode, omit for byte-level).
        #[arg(short, long)]
        tokenizer: Option<String>,
        /// Path to val.bin for perplexity
        #[arg(long, default_value = "data/val.bin")]
        val_data: String,
        /// Directory containing benchmark .txt files
        #[arg(long, default_value = "data")]
        data_dir: String,
    },
}

fn select_device(cpu: bool) -> Device {
    if cpu {
        eprintln!("Using CPU");
        Device::Cpu
    } else {
        match Device::new_metal(0) {
            Ok(d) => { eprintln!("Using Metal GPU"); d }
            Err(e) => { eprintln!("Metal not available ({}), falling back to CPU", e); Device::Cpu }
        }
    }
}

fn load_model(model_path: &str, config_path: &str, device: &Device) -> anyhow::Result<(NmModel, ModelConfig)> {
    eprintln!("Loading config: {}", config_path);
    let mut config: ModelConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    if config.byte_level {
        config.vocab_size = 259;
    }
    eprintln!("Model: d_model={}, layers={}, heads={}, vocab={}{}",
        config.d_model, config.n_layers, config.n_heads, config.vocab_size,
        if config.byte_level { " (byte-level)" } else { "" });

    eprintln!("Loading model: {}", model_path);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)?
    };
    let model = NmModel::load(vb.pp("item"), &config)?;
    eprintln!("Model loaded.");
    Ok((model, config))
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let device = select_device(cli.cpu);

    match cli.command {
        Command::Generate { model, config, tokenizer, prompt, max_tokens, temperature } => {
            let (model, model_config) = load_model(&model, &config, &device)?;

            // Encode prompt and set up decode function based on mode
            let (prompt_ids, eos_id, tok): (Vec<u32>, u32, Option<tokenizers::Tokenizer>) =
                if model_config.byte_level {
                    let ids: Vec<u32> = prompt.as_bytes().iter().map(|&b| b as u32 + 3).collect();
                    (ids, 2u32, None)
                } else {
                    let tok_path = tokenizer.as_deref().unwrap_or("tokenizer.json");
                    let tok = tokenizers::Tokenizer::from_file(tok_path)
                        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
                    let encoding = tok.encode(prompt.as_str(), false)
                        .map_err(|e| anyhow::anyhow!("Failed to encode: {}", e))?;
                    let ids = encoding.get_ids().to_vec();
                    let eos = tok.token_to_id("<eos>").unwrap_or(2);
                    (ids, eos, Some(tok))
                };

            eprintln!("Prompt: {} tokens{}", prompt_ids.len(),
                if model_config.byte_level { " (byte-level)" } else { "" });

            let mut states = model.init_states(&device)?;

            // Process prompt
            let start = std::time::Instant::now();
            let mut last_logits = None;
            for &token_id in &prompt_ids {
                last_logits = Some(model.forward_step(token_id, &mut states, &device)?);
            }
            let prompt_time = start.elapsed();
            eprintln!("Prompt processed in {:.1}ms ({:.0} tok/s)",
                prompt_time.as_millis(),
                prompt_ids.len() as f64 / prompt_time.as_secs_f64());

            // Generate
            print!("{}", prompt);
            let mut generated = Vec::new();
            let gen_start = std::time::Instant::now();

            for _ in 0..max_tokens {
                let logits = match &last_logits {
                    Some(l) => l.clone(),
                    None => break,
                };

                let next_token = if temperature <= 0.0 {
                    logits.argmax(0)?.to_scalar::<u32>()?
                } else {
                    let scaled = (&logits / temperature)?;
                    let probs = candle_nn::ops::softmax(&scaled, 0)?;
                    sample_from_probs(&probs)?
                };

                if next_token == eos_id { break; }
                generated.push(next_token);

                // Decode and print
                if model_config.byte_level {
                    if next_token >= 3 && next_token < 259 {
                        let byte = (next_token - 3) as u8;
                        if let Ok(ch) = std::str::from_utf8(&[byte]) {
                            print!("{}", ch);
                        }
                    }
                } else if let Some(ref tok) = tok {
                    if let Ok(text) = tok.decode(&[next_token], true) {
                        print!("{}", text);
                    }
                }
                use std::io::Write;
                std::io::stdout().flush()?;

                last_logits = Some(model.forward_step(next_token, &mut states, &device)?);
            }

            let gen_time = gen_start.elapsed();
            println!();
            eprintln!("\nGenerated {} tokens in {:.1}ms ({:.1} tok/s)",
                generated.len(), gen_time.as_millis(),
                generated.len() as f64 / gen_time.as_secs_f64());
        }

        Command::Evaluate { model, config, tokenizer, val_data, data_dir } => {
            let (model, config) = load_model(&model, &config, &device)?;

            let tok = if config.byte_level {
                None
            } else {
                let tok_path = tokenizer.as_deref().unwrap_or("tokenizer.json");
                Some(tokenizers::Tokenizer::from_file(tok_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?)
            };

            eval::run_eval(&model, &config, tok.as_ref(), &device, &val_data, &data_dir);
        }
    }

    Ok(())
}

fn sample_from_probs(probs: &Tensor) -> candle_core::Result<u32> {
    let probs_vec: Vec<f32> = probs.to_vec1()?;
    let mut rng = rand::thread_rng();
    let r: f32 = rand::Rng::gen(&mut rng);
    let mut cumsum = 0.0f32;
    for (i, &p) in probs_vec.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return Ok(i as u32);
        }
    }
    Ok((probs_vec.len() - 1) as u32)
}
