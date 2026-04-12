/// Data pipeline for training.
///
/// Supports:
/// - Pre-tokenization to binary format (u32 little-endian)
/// - Binary token dataset loading

use std::fs;
use std::io::{BufRead, BufReader, Write};

use crate::tokenizer::Tokenizer;

/// Pre-tokenize a text file to binary u32 token format.
pub fn preprocess(input_path: &str, output_path: &str, tokenizer_path: &str) {
    let tokenizer = Tokenizer::from_file(tokenizer_path);
    let file = fs::File::open(input_path).expect("failed to open input file");
    let reader = BufReader::with_capacity(1 << 20, file);

    let mut tokens: Vec<u32> = Vec::new();
    let mut lines_read = 0u64;

    let bos = tokenizer.token_to_id("<bos>").unwrap_or(1);
    let eos = tokenizer.token_to_id("<eos>").unwrap_or(2);

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        tokens.push(bos);
        tokens.extend_from_slice(&tokenizer.encode(line));
        tokens.push(eos);

        lines_read += 1;
        if lines_read % 100_000 == 0 {
            eprintln!("  tokenized {}k lines, {} tokens...", lines_read / 1000, tokens.len());
        }
    }

    // Write as little-endian u32
    let mut out = fs::File::create(output_path).expect("failed to create output file");
    for &t in &tokens {
        out.write_all(&t.to_le_bytes()).expect("failed to write");
    }

    eprintln!("Preprocessed {} lines → {} tokens → {}", lines_read, tokens.len(), output_path);
}

/// Binary token dataset for training.
pub struct TokenDataset {
    pub tokens: Vec<u32>,
    pub seq_len: usize,
}

impl TokenDataset {
    /// Load from a binary file of u32 tokens.
    pub fn from_binary(path: &str, seq_len: usize) -> Self {
        let bytes = fs::read(path).expect("failed to read binary token file");
        assert!(bytes.len() % 4 == 0, "binary file size must be multiple of 4");
        let tokens: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        eprintln!("Loaded {} tokens from {}", tokens.len(), path);
        Self { tokens, seq_len }
    }

    /// Number of possible windows.
    pub fn len(&self) -> usize {
        if self.tokens.len() <= self.seq_len {
            0
        } else {
            self.tokens.len() - self.seq_len
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Find a large prime coprime with n, for stride-based sequential sampling.
/// Guarantees visiting every index exactly once in n steps: idx = (pos * stride) % n
pub fn find_coprime_stride(n: usize) -> usize {
    // Use a large prime that's coprime with any n (since it's prime, coprime unless n is a multiple)
    // Pick different primes to avoid pathological patterns
    let primes = [
        104_729, 224_737, 350_377, 479_001, 611_953,
        746_773, 882_377, 999_979, 1_200_007, 1_500_007,
    ];
    for &p in &primes {
        if n % p != 0 {
            return p;
        }
    }
    // Fallback: any odd number coprime with n
    let mut s = 1_000_003;
    while gcd(s, n) != 1 {
        s += 2;
    }
    s
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
