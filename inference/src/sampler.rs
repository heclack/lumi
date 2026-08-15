/// Host-side sampling from a logits vector: greedy (temperature 0), or
/// temperature softmax with optional top-k truncation. Mirrors the sampling
/// logic in `archive/metal-inference/src/main.rs` (argmax / softmax + inverse-
/// CDF draw), generalized with a top-k cutoff and a seeded RNG for
/// reproducible generation.
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct Sampler {
    rng: StdRng,
    temperature: f64,
    top_k: Option<usize>,
}

impl Sampler {
    pub fn new(temperature: f64, top_k: Option<usize>, seed: u64) -> Self {
        Self { rng: StdRng::seed_from_u64(seed), temperature, top_k }
    }

    /// Pick the next token id from a logits vector of length `vocab_size`.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return argmax(logits);
        }

        let mut scaled: Vec<f32> =
            logits.iter().map(|&l| l / self.temperature as f32).collect();

        // Top-k: zero out (via -inf) every logit outside the k highest, so the
        // softmax below assigns them no probability mass. The threshold is the
        // k-th largest value — O(V) selection, no need to sort the whole vocab.
        if let Some(k) = self.top_k {
            if k > 0 && k < scaled.len() {
                let mut sel = scaled.clone();
                let (_, kth, _) = sel.select_nth_unstable_by(k - 1, |a, b| {
                    b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
                });
                let threshold = *kth;
                for v in scaled.iter_mut() {
                    if *v < threshold {
                        *v = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Softmax in place: `scaled` becomes the (unnormalized) exp weights.
        let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in scaled.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }

        // Inverse-CDF sampling over the (renormalized) softmax distribution.
        let r: f32 = self.rng.gen_range(0.0..1.0);
        let mut cumsum = 0.0f32;
        for (i, &e) in scaled.iter().enumerate() {
            cumsum += e / sum;
            if cumsum >= r {
                return i as u32;
            }
        }
        (scaled.len() - 1) as u32
    }
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}
