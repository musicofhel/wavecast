use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::collections::HashMap;

/// Build a bag-of-words frequency counter from a list of words.
#[pyfunction]
pub fn build_bow_rs(words: Vec<String>) -> PyResult<HashMap<String, usize>> {
    let mut map: HashMap<String, usize> = HashMap::new();
    for w in words {
        *map.entry(w).or_insert(0) += 1;
    }
    Ok(map)
}

/// Build a TF-IDF matrix from a list of bag-of-words dictionaries.
///
/// Returns `(tfidf_matrix, vocabulary)` where:
/// - `tfidf_matrix` is a 2-D numpy float64 array of shape `(n_docs, vocab_size)`
/// - `vocabulary` is the sorted list of unique words
#[pyfunction]
pub fn build_corpus_tfidf_rs(
    py: Python<'_>,
    bow_list: Vec<HashMap<String, usize>>,
) -> PyResult<(Py<PyArray2<f64>>, Vec<String>)> {
    if bow_list.is_empty() {
        let arr = PyArray2::<f64>::zeros(py, [0, 0], false);
        return Ok((arr.unbind(), Vec::new()));
    }

    // Build sorted vocabulary.
    let mut vocab_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for bow in &bow_list {
        for key in bow.keys() {
            vocab_set.insert(key.clone());
        }
    }
    let vocabulary: Vec<String> = vocab_set.into_iter().collect();
    let n_vocab = vocabulary.len();
    let n_docs = bow_list.len();

    if n_vocab == 0 {
        let arr = PyArray2::<f64>::zeros(py, [n_docs, 0], false);
        return Ok((arr.unbind(), Vec::new()));
    }

    let vocab_index: HashMap<&str, usize> = vocabulary
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i))
        .collect();

    // Compute TF matrix.
    let mut tf = vec![0.0f64; n_docs * n_vocab];
    for (d, bow) in bow_list.iter().enumerate() {
        let total: usize = bow.values().sum();
        if total > 0 {
            let total_f = total as f64;
            for (word, &count) in bow {
                if let Some(&idx) = vocab_index.get(word.as_str()) {
                    tf[d * n_vocab + idx] = count as f64 / total_f;
                }
            }
        }
    }

    // Document frequency and IDF.
    let mut df = vec![0.0f64; n_vocab];
    for bow in &bow_list {
        for word in bow.keys() {
            if let Some(&idx) = vocab_index.get(word.as_str()) {
                df[idx] += 1.0;
            }
        }
    }

    let n_docs_f = n_docs as f64;
    let idf: Vec<f64> = df
        .iter()
        .map(|&d| {
            if d > 0.0 {
                n_docs_f.ln() - d.ln()
            } else {
                n_docs_f.ln()
            }
        })
        .collect();

    // TF-IDF = TF * IDF.
    for d in 0..n_docs {
        for v in 0..n_vocab {
            tf[d * n_vocab + v] *= idf[v];
        }
    }

    // Create numpy array from the flat vec.
    let arr = PyArray2::<f64>::zeros(py, [n_docs, n_vocab], false);
    unsafe {
        let slice = arr.as_slice_mut().unwrap();
        slice.copy_from_slice(&tf);
    }
    Ok((arr.unbind(), vocabulary))
}
