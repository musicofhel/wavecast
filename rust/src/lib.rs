mod bow;
mod dataset;
mod sax_words;
mod vocab;

use pyo3::prelude::*;

/// Native acceleration module for WaveCast hot paths.
#[pymodule]
fn _wavecast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sax_words::extract_words_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bow::build_bow_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bow::build_corpus_tfidf_rs, m)?)?;
    m.add_function(wrap_pyfunction!(vocab::encode_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dataset::build_sliding_windows_rs, m)?)?;
    Ok(())
}
