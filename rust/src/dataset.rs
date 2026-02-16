use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Build sliding-window (context, target) pairs from a 1-D token array.
///
/// Returns `(contexts, targets)` as 2-D i64 numpy arrays:
/// - `contexts`: shape `(n_windows, context_length)`
/// - `targets`: shape `(n_windows, max_horizon)`
///
/// For each position `i` in `0..n_windows`:
///   context = tokens[i .. i+context_length]
///   targets[j] = tokens[i+context_length+j]  for j in 0..max_horizon
#[pyfunction]
pub fn build_sliding_windows_rs(
    py: Python<'_>,
    tokens: Vec<i64>,
    context_length: usize,
    max_horizon: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<i64>>)> {
    let n = tokens.len();
    let needed = context_length + max_horizon;
    if n < needed {
        let ctx = PyArray2::<i64>::zeros(py, [0, context_length], false);
        let tgt = PyArray2::<i64>::zeros(py, [0, max_horizon], false);
        return Ok((ctx.unbind(), tgt.unbind()));
    }

    let n_windows = n - needed + 1;

    let ctx_arr = PyArray2::<i64>::zeros(py, [n_windows, context_length], false);
    let tgt_arr = PyArray2::<i64>::zeros(py, [n_windows, max_horizon], false);

    unsafe {
        let ctx_slice = ctx_arr.as_slice_mut().unwrap();
        let tgt_slice = tgt_arr.as_slice_mut().unwrap();

        for i in 0..n_windows {
            // Copy context window.
            ctx_slice[i * context_length..(i + 1) * context_length]
                .copy_from_slice(&tokens[i..i + context_length]);
            // Copy target horizons.
            for h in 0..max_horizon {
                tgt_slice[i * max_horizon + h] = tokens[i + context_length + h];
            }
        }
    }

    Ok((ctx_arr.unbind(), tgt_arr.unbind()))
}
