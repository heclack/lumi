/// Lumi — hybrid Mamba-3 language model CLI.
///
/// Subcommands:
///   train-tokenizer  Train a BPE tokenizer on corpus
///   preprocess        Tokenize corpus to binary format
///   train             Train the model (native CUDA)
///   smoke-test        Verify CUDA kernels
///
/// For evaluation and text generation, use the inference binary: lumi-infer

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "lumi", about = "Lumi — hybrid Mamba-3 language model")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Train a custom BPE tokenizer on corpus file(s).
    TrainTokenizer {
        /// Input corpus files (comma-separated).
        #[arg(short, long)]
        corpus: String,
        /// Target vocabulary size.
        #[arg(short, long, default_value = "32000")]
        vocab_size: usize,
        /// Output tokenizer JSON path.
        #[arg(short, long, default_value = "tokenizer.json")]
        output: String,
    },
    /// Pre-tokenize corpus to binary token file.
    Preprocess {
        /// Input text file.
        #[arg(short, long)]
        input: String,
        /// Output binary file.
        #[arg(short, long)]
        output: String,
        /// Tokenizer JSON path (required for BPE mode, ignored for byte-level).
        #[arg(short, long)]
        tokenizer: Option<String>,
        /// Use byte-level encoding (no tokenizer needed).
        #[arg(long)]
        byte_level: bool,
    },
    /// Train the model (native CUDA).
    Train {
        /// Config JSON path (optional, uses defaults if not provided).
        #[arg(short, long)]
        config: Option<String>,
        /// CUDA device ID.
        #[arg(short = 'd', long, default_value = "0")]
        device: i32,
    },
    /// Smoke test CUDA kernels (forward pass).
    SmokeTest {
        /// Config JSON path (optional).
        #[arg(short, long)]
        config: Option<String>,
        /// Run full forward pass for all config variants (pure Mamba, hybrid, byte-level, etc.).
        #[arg(long)]
        all_configs: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::TrainTokenizer { corpus, vocab_size, output } => {
            let paths: Vec<&str> = corpus.split(',').collect();
            lumi::tokenizer::Tokenizer::train_bpe(&paths, vocab_size, &output);
        }
        Command::Preprocess { input, output, tokenizer, byte_level } => {
            if byte_level {
                lumi::data::preprocess_bytes(&input, &output);
            } else {
                let tokenizer_path = tokenizer.unwrap_or_else(|| "tokenizer.json".to_string());
                lumi::data::preprocess(&input, &output, &tokenizer_path);
            }
        }
        Command::Train { config, device } => {
            let config = match config {
                Some(path) => lumi::config::TrainingConfig::from_file(&path)
                    .expect("failed to load config"),
                None => lumi::config::TrainingConfig::default(),
            };

            eprintln!("=== Lumi — Native CUDA Training — Mamba-3 {}M ===",
                config.model.param_count() / 1_000_000);
            lumi::native_trainer::train_native(&config, device);
        }
        Command::SmokeTest { config: config_path, all_configs } => {
            #[cfg(feature = "cuda")]
            {
                if all_configs {
                    lumi::native_trainer::smoke_test_configs();
                } else {
                    let config = match config_path {
                        Some(path) => lumi::config::TrainingConfig::from_file(&path)
                            .expect("failed to load config"),
                        None => lumi::config::TrainingConfig::default(),
                    };
                    lumi::native_trainer::smoke_test_forward(&config);
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = config_path;
                let _ = all_configs;
                eprintln!("smoke-test requires --features cuda");
            }
        }
    }
}
