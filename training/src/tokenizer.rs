/// Tokenizer wrapper using HuggingFace tokenizers crate.
///
/// Supports training a custom BPE on the corpus and loading from file.

use tokenizers::models::bpe::{BPE, BpeTrainer};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelProcessor;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::AddedToken;
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a JSON file.
    pub fn from_file(path: &str) -> Self {
        let inner = HfTokenizer::from_file(path).expect("failed to load tokenizer");
        Self { inner }
    }

    /// Train a BPE tokenizer on corpus file(s).
    pub fn train_bpe(corpus_paths: &[&str], vocab_size: usize, output_path: &str) {
        let special_tokens = vec![
            AddedToken::from("<pad>", true),
            AddedToken::from("<bos>", true),
            AddedToken::from("<eos>", true),
            AddedToken::from("<unk>", true),
        ];

        let trainer = BpeTrainer::builder()
            .vocab_size(vocab_size)
            .special_tokens(special_tokens)
            .show_progress(true)
            .build();

        let mut tokenizer = HfTokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        tokenizer.with_post_processor(Some(ByteLevelProcessor::default()));

        let paths: Vec<String> = corpus_paths.iter().map(|p| p.to_string()).collect();
        tokenizer
            .train_from_files(&mut TrainerWrapper::from(trainer), paths)
            .expect("failed to train tokenizer");

        // Save as the general Tokenizer type
        tokenizer
            .save(output_path, true)
            .expect("failed to save tokenizer");

        eprintln!("Tokenizer saved to {output_path} (vocab_size: {vocab_size})");
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.inner.encode(text, false).expect("failed to encode");
        encoding.get_ids().to_vec()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens, true).unwrap_or_default()
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get token ID for a special token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
