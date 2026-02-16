use pyo3::prelude::*;
use std::collections::HashMap;

/// Batch-encode a list of words to token IDs using a word→id map.
///
/// Unknown words are mapped to `unk_id`.
#[pyfunction]
pub fn encode_batch_rs(
    words: Vec<String>,
    word_to_id: HashMap<String, i64>,
    unk_id: i64,
) -> PyResult<Vec<i64>> {
    let result: Vec<i64> = words
        .iter()
        .map(|w| *word_to_id.get(w).unwrap_or(&unk_id))
        .collect();
    Ok(result)
}
