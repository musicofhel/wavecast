use pyo3::prelude::*;

/// Extract SAX words using a sliding window over a symbol string.
///
/// Equivalent to the Python `extract_words()` but runs without the GIL.
#[pyfunction]
pub fn extract_words_rs(
    py: Python<'_>,
    symbols: &str,
    word_length: usize,
    stride: usize,
) -> PyResult<Vec<String>> {
    if word_length == 0 || stride == 0 {
        return Ok(Vec::new());
    }
    let len = symbols.len();
    if len < word_length {
        return Ok(Vec::new());
    }

    // Release the GIL for the pure-Rust computation.
    let symbols_owned = symbols.to_owned();
    let words = py.allow_threads(move || {
        let bytes = symbols_owned.as_bytes();
        let n = if len >= word_length {
            (len - word_length) / stride + 1
        } else {
            0
        };
        let mut result = Vec::with_capacity(n);
        let mut i = 0;
        while i + word_length <= bytes.len() {
            // SAX symbols are ASCII, so byte slicing is safe.
            result.push(
                std::str::from_utf8(&bytes[i..i + word_length])
                    .unwrap()
                    .to_owned(),
            );
            i += stride;
        }
        result
    });
    Ok(words)
}
