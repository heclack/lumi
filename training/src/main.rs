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
        /// Tokenizer JSON path.
        #[arg(short, long, default_value = "tokenizer.json")]
        tokenizer: String,
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
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::TrainTokenizer { corpus, vocab_size, output } => {
            let paths: Vec<&str> = corpus.split(',').collect();
            lumi::tokenizer::Tokenizer::train_bpe(&paths, vocab_size, &output);
        }
        Command::Preprocess { input, output, tokenizer } => {
            lumi::data::preprocess(&input, &output, &tokenizer);
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
        Command::SmokeTest { config: config_path } => {
            #[cfg(feature = "cuda")]
            {
                let config = match config_path {
                    Some(path) => lumi::config::TrainingConfig::from_file(&path)
                        .expect("failed to load config"),
                    None => lumi::config::TrainingConfig::default(),
                };
                lumi::native_trainer::smoke_test_forward(&config);
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = config_path;
                eprintln!("smoke-test requires --features cuda");
            }
        }
    }
}
