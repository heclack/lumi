/// Evaluation harness for Candle inference.
///
/// Benchmarks:
///   - Perplexity on held-out data
///   - Multiple-choice accuracy (ARC-Easy, BoolQ, WinoGrande)
///   - Generation quality (fixed prompts)

use candle_core::{Device, IndexOp, Result};
use crate::model::{NmModel, ModelConfig};

/// A single multiple-choice question.
pub struct McQuestion {
    pub context: Vec<u32>,
    pub choices: Vec<Vec<u32>>,
    pub label: usize,
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub accuracy: f32,
    pub total: usize,
    pub correct: usize,
}

#[derive(Debug)]
pub struct PerplexityResult {
    pub name: String,
    pub perplexity: f32,
    pub avg_loss: f32,
    pub num_tokens: usize,
}

/// Compute perplexity on a binary token file (u32 little-endian).
///
/// Runs `NmModel::forward_window` once per window. The windowed kernel batches
/// projections and scans the SSM in a single Metal dispatch, so this is both
/// fast and numerically equivalent to `forward_step`.
pub fn evaluate_perplexity(
    model: &NmModel,
    token_file: &str,
    seq_len: usize,
    max_windows: usize,
    device: &Device,
) -> Result<PerplexityResult> {
    let name = std::path::Path::new(token_file)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| token_file.to_string());

    let bytes = std::fs::read(token_file).expect("failed to read token file");
    let tokens: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    if tokens.len() <= seq_len {
        return Ok(PerplexityResult { name, perplexity: f32::INFINITY, avg_loss: f32::INFINITY, num_tokens: 0 });
    }

    let stride = seq_len;
    let n_eval = ((tokens.len() - seq_len) / stride).min(max_windows);

    let mut total_loss = 0.0f64;
    let mut total_tokens = 0usize;

    for i in 0..n_eval {
        let start = i * stride;
        if start + seq_len >= tokens.len() { break; }

        let window = &tokens[start..start + seq_len + 1];

        // Fresh state per window — perplexity windows are independent.
        let mut states = model.init_states(device)?;

        // Single windowed forward → [seq_len, vocab_size]
        let logits = model.forward_window(&window[..seq_len], &mut states, device)?;
        let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;

        for pos in 0..seq_len {
            let target = window[pos + 1] as usize;
            let lp: f32 = log_probs.i(pos)?.i(target)?.to_scalar()?;
            total_loss -= lp as f64;
            total_tokens += 1;
        }

        if (i + 1) % 10 == 0 {
            let running_ppl = (total_loss / total_tokens as f64).exp();
            eprintln!("  [{}/{}] running ppl: {:.2}", i + 1, n_eval, running_ppl);
        }
    }

    let avg_loss = total_loss / total_tokens.max(1) as f64;
    let perplexity = avg_loss.exp() as f32;

    Ok(PerplexityResult { name, perplexity, avg_loss: avg_loss as f32, num_tokens: total_tokens })
}

/// Score a continuation by summing log-probs of target tokens given context.
///
/// Runs `forward_window` once on the concatenated `context + continuation`
/// sequence, then reads the per-position log-probs at the continuation indices.
/// Returns mean log-prob per continuation token (length-normalized).
fn score_continuation(
    model: &NmModel,
    context: &[u32],
    continuation: &[u32],
    device: &Device,
    max_seq_len: usize,
) -> Result<f32> {
    if continuation.is_empty() {
        return Ok(f32::NEG_INFINITY);
    }

    // Left-trim to max_seq_len when too long: most recent context wins.
    let mut full: Vec<u32> = Vec::with_capacity(context.len() + continuation.len());
    full.extend_from_slice(context);
    full.extend_from_slice(continuation);
    let seq_len = full.len().min(max_seq_len);
    let full = &full[full.len().saturating_sub(seq_len)..];
    let ctx_len = full.len() - continuation.len();

    // Need at least one context token so that log_probs[ctx_len - 1] exists.
    if ctx_len == 0 {
        return Ok(f32::NEG_INFINITY);
    }

    let mut states = model.init_states(device)?;
    let logits = model.forward_window(full, &mut states, device)?;   // [len, vocab]
    let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;

    // log_probs[pos - 1] predicts full[pos]; sum over the continuation slice.
    let mut total_lp = 0.0f32;
    for pos in ctx_len..full.len() {
        let lp: f32 = log_probs.i(pos - 1)?.i(full[pos] as usize)?.to_scalar()?;
        total_lp += lp;
    }

    Ok(total_lp / continuation.len() as f32)
}

/// Evaluate multiple-choice questions.
pub fn evaluate_mc(
    model: &NmModel,
    name: &str,
    questions: &[McQuestion],
    device: &Device,
    max_seq_len: usize,
) -> Result<BenchmarkResult> {
    let total = questions.len();
    let mut correct = 0usize;

    for (i, q) in questions.iter().enumerate() {
        let scores: Vec<f32> = q.choices.iter()
            .map(|choice| score_continuation(model, &q.context, choice, device, max_seq_len).unwrap_or(f32::NEG_INFINITY))
            .collect();

        let predicted = scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        if predicted == q.label {
            correct += 1;
        }

        if (i + 1) % 100 == 0 || i + 1 == total {
            eprintln!("  [{}/{}] acc: {:.1}%", i + 1, total, correct as f32 / (i + 1) as f32 * 100.0);
        }
    }

    Ok(BenchmarkResult {
        name: name.to_string(),
        accuracy: correct as f32 / total.max(1) as f32,
        total,
        correct,
    })
}

// ─── Benchmark Loaders ───────────────────────────────────────

/// Byte-level encoding: each UTF-8 byte maps to ID = byte_value + 3.
fn encode_bytes(text: &str) -> Vec<u32> {
    text.as_bytes().iter().map(|&b| b as u32 + 3).collect()
}

/// Build an encode function from an optional tokenizer. Uses byte-level if None.
fn make_encode_fn(tokenizer: Option<&tokenizers::Tokenizer>) -> Box<dyn Fn(&str) -> Vec<u32> + '_> {
    match tokenizer {
        Some(tok) => Box::new(move |text: &str| {
            tok.encode(text, false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default()
        }),
        None => Box::new(encode_bytes),
    }
}

// Embedded benchmark snapshots — decoupled from the mutable training data corpus.
const ARC_EASY_DATA: &str = include_str!("../benchmarks/arc_easy.txt");
const WINOGRANDE_DATA: &str = include_str!("../benchmarks/winogrande.txt");

pub fn load_arc_easy(encode_fn: &dyn Fn(&str) -> Vec<u32>, max_questions: usize) -> Vec<McQuestion> {
    // One "question? answer" per line. Split at the last '?' — prefix is context, suffix is the
    // correct answer. Distractors are drawn from other lines' answers.
    let pairs: Vec<(String, String)> = ARC_EASY_DATA.lines()
        .filter_map(|line| {
            let line = line.trim();
            let idx = line.rfind('?')?;
            let ctx = line[..=idx].trim().to_string();
            let ans = line[idx + 1..].trim().to_string();
            if ctx.is_empty() || ans.is_empty() { None } else { Some((ctx, ans)) }
        })
        .collect();

    if pairs.len() < 4 { return Vec::new(); }

    let all_answers: Vec<Vec<u32>> = pairs.iter().map(|(_, a)| encode_fn(&format!(" {}", a))).collect();

    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;
    let mut questions = Vec::new();

    for (idx, (ctx, _)) in pairs.iter().enumerate() {
        if questions.len() >= max_questions { break; }
        let context = encode_fn(ctx);
        let correct = all_answers[idx].clone();

        let mut distractor_indices: Vec<usize> = (0..all_answers.len()).filter(|&i| i != idx).collect();
        distractor_indices.shuffle(&mut rng);

        let mut choice_pairs: Vec<(Vec<u32>, bool)> = vec![(correct, true)];
        for &di in distractor_indices.iter().take(3) {
            choice_pairs.push((all_answers[di].clone(), false));
        }
        choice_pairs.shuffle(&mut rng);

        let label = choice_pairs.iter().position(|(_, c)| *c).unwrap_or(0);
        let choices: Vec<Vec<u32>> = choice_pairs.into_iter().map(|(c, _)| c).collect();
        questions.push(McQuestion { context, choices, label });
    }
    questions
}

pub fn load_winogrande(encode_fn: &dyn Fn(&str) -> Vec<u32>, max_questions: usize) -> Vec<McQuestion> {
    // Consecutive line pairs share a prefix; they diverge where the answer starts. Line 1 is the
    // correct continuation, line 2 is the distractor. Score = log-prob of the divergent tail given
    // the common prefix.
    let lines: Vec<&str> = WINOGRANDE_DATA.lines().filter(|l| !l.trim().is_empty()).collect();
    let mut questions = Vec::new();

    for pair in lines.chunks_exact(2) {
        if questions.len() >= max_questions { break; }
        let a = pair[0].trim();
        let b = pair[1].trim();

        // Find common prefix length (in bytes, at char boundary).
        let mut prefix_end = 0;
        for ((i, ca), cb) in a.char_indices().zip(b.chars()) {
            if ca != cb { break; }
            prefix_end = i + ca.len_utf8();
        }
        // Back up to the last space so we split cleanly on a word boundary.
        while prefix_end > 0 && !a.is_char_boundary(prefix_end) {
            prefix_end -= 1;
        }
        if let Some(space_idx) = a[..prefix_end].rfind(' ') {
            prefix_end = space_idx;
        }
        if prefix_end == 0 { continue; }

        let context = encode_fn(a[..prefix_end].trim_end());
        let tail_a = encode_fn(&a[prefix_end..]);
        let tail_b = encode_fn(&b[prefix_end..]);
        if tail_a.is_empty() || tail_b.is_empty() { continue; }

        questions.push(McQuestion { context, choices: vec![tail_a, tail_b], label: 0 });
    }
    questions
}

pub fn generation_prompts() -> Vec<(&'static str, &'static str)> {
    vec![
        ("story_start", "Once upon a time, a little girl named Lily"),
        ("conversation", "Mom: Good morning! How did you sleep?\nChild:"),
        ("science", "The sun is a star that"),
        ("math_word", "If you have 5 apples and give 2 to your friend, you have"),
        ("nature", "Butterflies start their life as"),
        ("social", "When someone is feeling sad, you can"),
        ("animal", "Dogs are friendly animals that"),
        ("instruction", "To make a peanut butter sandwich, first"),
        ("rhyme", "Twinkle twinkle little star,"),
        ("question", "Why is the sky blue?"),
    ]
}

/// Run the full evaluation suite.
pub fn run_eval(
    model: &NmModel,
    config: &ModelConfig,
    tokenizer: Option<&tokenizers::Tokenizer>,
    device: &Device,
    val_data: &str,
) {
    eprintln!("=== Lumi Evaluation Suite{} ===\n",
        if config.byte_level { " (byte-level)" } else { "" });

    let encode_fn = make_encode_fn(tokenizer);

    // --- Perplexity ---
    eprintln!("--- Perplexity ---");
    if std::path::Path::new(val_data).exists() {
        match evaluate_perplexity(model, val_data, config.max_seq_len.min(512), 50, device) {
            Ok(ppl) => eprintln!("  {} perplexity: {:.2} (avg loss: {:.4}, {} tokens)\n",
                ppl.name, ppl.perplexity, ppl.avg_loss, ppl.num_tokens),
            Err(e) => eprintln!("  Perplexity error: {}\n", e),
        }
    } else {
        eprintln!("  {} not found, skipping\n", val_data);
    }

    // --- Multiple Choice (benchmarks are embedded in the binary) ---
    let mc_max = 200;
    let mc_seq_len = config.max_seq_len.min(512);

    let benchmarks: Vec<(&str, Vec<McQuestion>)> = vec![
        ("ARC-Easy", load_arc_easy(&*encode_fn, mc_max)),
        ("WinoGrande", load_winogrande(&*encode_fn, mc_max)),
    ];

    for (name, questions) in &benchmarks {
        eprintln!("--- {} ---", name);
        if questions.is_empty() {
            eprintln!("  No questions loaded\n");
            continue;
        }
        match evaluate_mc(model, name, questions, device, mc_seq_len) {
            Ok(result) => eprintln!("  {}: {:.1}% ({}/{})\n",
                result.name, result.accuracy * 100.0, result.correct, result.total),
            Err(e) => eprintln!("  Error: {}\n", e),
        }
    }

    // --- Generation Prompts ---
    eprintln!("--- Generation Prompts ---");
    for (label, prompt) in generation_prompts() {
        eprintln!("  [{}] \"{}\"", label, prompt);
    }
    eprintln!("\nRun generation with: lumi-infer --model model.safetensors --prompt \"...\"");
}
