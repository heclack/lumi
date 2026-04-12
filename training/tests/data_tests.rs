/// Tests for data loading, batch sampling, and sequential stride.

use lumi::data::{TokenDataset, find_coprime_stride, preprocess_bytes};
use std::io::Write;

fn create_temp_dataset(tokens: &[u32]) -> (tempfile::NamedTempFile, TokenDataset) {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    for &t in tokens {
        f.write_all(&t.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
    let path = f.path().to_str().unwrap().to_string();
    let ds = TokenDataset::from_binary(&path, 4);
    (f, ds)
}

#[test]
fn dataset_load_and_len() {
    let tokens: Vec<u32> = (0..100).collect();
    let (_f, ds) = create_temp_dataset(&tokens);

    assert_eq!(ds.tokens.len(), 100);
    assert_eq!(ds.seq_len, 4);
    // Number of valid windows = tokens - seq_len = 96
    assert_eq!(ds.len(), 96);
}

#[test]
fn dataset_empty_if_too_short() {
    let tokens: Vec<u32> = vec![1, 2, 3];
    let (_f, ds) = create_temp_dataset(&tokens);

    // 3 tokens with seq_len=4 → no valid windows
    assert_eq!(ds.len(), 0);
    assert!(ds.is_empty());
}

#[test]
fn dataset_tokens_within_bounds() {
    let tokens: Vec<u32> = (0..50).collect();
    let (_f, ds) = create_temp_dataset(&tokens);

    // Every window [idx..idx+seq] and target [idx+1..idx+seq+1] should be valid
    let seq = ds.seq_len;
    for idx in 0..ds.len() {
        // Input window
        for j in 0..seq {
            assert!(idx + j < ds.tokens.len(), "OOB input at idx={}, j={}", idx, j);
        }
        // Target window (shifted by 1)
        for j in 0..seq {
            assert!(idx + j + 1 < ds.tokens.len(), "OOB target at idx={}, j={}", idx, j);
        }
    }
}

#[test]
fn sequential_stride_no_repeats() {
    let tokens: Vec<u32> = (0..200).collect();
    let (_f, ds) = create_temp_dataset(&tokens);
    let n = ds.len(); // 196
    let stride = find_coprime_stride(n);

    let mut visited = vec![false; n];
    for i in 0..n {
        let idx = (i * stride) % n;
        assert!(!visited[idx], "Duplicate at pos {} (i={}, stride={}, n={})", idx, i, stride, n);
        visited[idx] = true;
    }
    assert!(visited.iter().all(|&v| v));
}

#[test]
fn gcd_helper_works() {
    // find_coprime_stride returns a value coprime with n
    for n in [10, 100, 256, 1000, 93170931] {
        let stride = find_coprime_stride(n);
        assert!(n % stride != 0, "stride {} divides n={}", stride, n);
        // Full coprimality: gcd(stride, n) == 1
        let mut a = stride;
        let mut b = n;
        while b != 0 { let t = b; b = a % b; a = t; }
        assert_eq!(a, 1, "gcd({}, {}) = {}, not coprime", stride, n, a);
    }
}

#[test]
fn preprocess_bytes_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let input_path = dir.path().join("input.txt");
    let output_path = dir.path().join("output.bin");

    std::fs::write(&input_path, "Hello\nWorld\n").unwrap();
    preprocess_bytes(input_path.to_str().unwrap(), output_path.to_str().unwrap());

    let bytes = std::fs::read(&output_path).unwrap();
    let tokens: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // "Hello" = [72, 101, 108, 108, 111] + 3 = [75, 104, 111, 111, 114]
    // "World" = [87, 111, 114, 108, 100] + 3 = [90, 114, 117, 111, 103]
    // With bos=1, eos=2 wrapping each line
    assert_eq!(tokens[0], 1); // bos
    assert_eq!(tokens[1], b'H' as u32 + 3);
    assert_eq!(tokens[2], b'e' as u32 + 3);
    assert_eq!(tokens[3], b'l' as u32 + 3);
    assert_eq!(tokens[4], b'l' as u32 + 3);
    assert_eq!(tokens[5], b'o' as u32 + 3);
    assert_eq!(tokens[6], 2); // eos
    assert_eq!(tokens[7], 1); // bos
    assert_eq!(tokens[8], b'W' as u32 + 3);
    assert_eq!(tokens[12], b'd' as u32 + 3);
    assert_eq!(tokens[13], 2); // eos
    assert_eq!(tokens.len(), 14); // 2 * (1 + 5 + 1)
}

#[test]
fn preprocess_bytes_skips_empty_lines() {
    let dir = tempfile::tempdir().unwrap();
    let input_path = dir.path().join("input.txt");
    let output_path = dir.path().join("output.bin");

    std::fs::write(&input_path, "Hi\n\n\nBye\n").unwrap();
    preprocess_bytes(input_path.to_str().unwrap(), output_path.to_str().unwrap());

    let bytes = std::fs::read(&output_path).unwrap();
    let tokens: Vec<u32> = bytes.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // "Hi" (2 bytes) + "Bye" (3 bytes) = 2*(bos+eos) + 5 = 9 tokens
    assert_eq!(tokens.len(), 9);
    assert_eq!(tokens[0], 1); // bos
    assert_eq!(tokens[3], 2); // eos after "Hi"
    assert_eq!(tokens[4], 1); // bos before "Bye"
    assert_eq!(tokens[8], 2); // eos after "Bye"
}
