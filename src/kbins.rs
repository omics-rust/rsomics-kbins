//! `KBinsDiscretizer` fit + transform for `uniform` and `quantile` strategies.
//!
//! Replicates `sklearn.preprocessing.KBinsDiscretizer` (1.9.0, BSD-3-Clause)
//! for the two deterministic strategies. The `kmeans` strategy is stochastic
//! and therefore excluded (see README). The `quantile` strategy uses
//! `averaged_inverted_cdf` (sklearn 1.9.0 default) or optionally `linear`
//! (NumPy type-7); see [`QuantileMethod`].
//!
//! ## Fit: `uniform`
//! `bin_edges[j] = np.linspace(col_min, col_max, n_bins+1)`.
//! Constant columns become a single bin spanning `(−∞, +∞)`.
//!
//! ## Fit: `quantile`
//! `percentile_levels = np.linspace(0, 100, n_bins+1)`.
//! Edges = `np.percentile(col, percentile_levels, method=<quantile_method>)`.
//! Consecutive-equal edges (≤ 1e-8 apart) are deduplicated (sklearn removes
//! bins whose width is too small with a warning; we replicate the dedup).
//! A near-constant column whose edges all collapse leaves a single edge —
//! sklearn reports `n_bins_ = 0` and maps every value to bin 0; we match that.
//! Exactly-constant columns are special-cased to a single `(−∞, +∞)` bin.
//!
//! ## Transform
//! `np.searchsorted(bin_edges[j][1:-1], x, side='right')` followed by
//! `np.clip(result, 0, n_bins_[j] - 1)`. Both inner edges strip the outer
//! fence values used only for `inverse_transform`.

use crate::quantile::{percentiles_aicdf, percentiles_linear};

/// Quantile method for the `quantile` strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// NumPy `averaged_inverted_cdf` (type-2, sklearn 1.9.0 default).
    AveragedInvertedCdf,
    /// NumPy `linear` (type-7). Matches sklearn with `quantile_method="linear"`.
    Linear,
}

/// Fitted per-column bin edges, ready for transform.
pub struct Fitted {
    /// Per-column effective n_bins (may be less than requested when edges collapse).
    pub n_bins: Vec<usize>,
    /// Per-column bin edges, including both outer fences.
    /// Length per column = `n_bins[j] + 1`.
    pub bin_edges: Vec<Vec<f64>>,
}

/// Bin strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Uniform,
    Quantile,
}

/// Fit bin edges for every column of the matrix.
///
/// # Errors
/// Returns an error if `n_bins < 2`.
pub fn fit(
    data: &[f64],
    n_rows: usize,
    n_cols: usize,
    n_bins: usize,
    strategy: Strategy,
    quantile_method: QuantileMethod,
) -> Fitted {
    assert!(n_bins >= 2, "n_bins must be at least 2");
    let mut fitted_n_bins = Vec::with_capacity(n_cols);
    let mut bin_edges = Vec::with_capacity(n_cols);

    for j in 0..n_cols {
        let col: Vec<f64> = (0..n_rows).map(|i| data[i * n_cols + j]).collect();

        let col_min = col.iter().copied().fold(f64::INFINITY, f64::min);
        let col_max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if col_min == col_max {
            // Constant column: single bin spanning (−∞, +∞), matching sklearn.
            fitted_n_bins.push(1);
            bin_edges.push(vec![f64::NEG_INFINITY, f64::INFINITY]);
            continue;
        }

        let edges = match strategy {
            Strategy::Uniform => linspace(col_min, col_max, n_bins + 1),
            Strategy::Quantile => {
                let levels = linspace(0.0, 100.0, n_bins + 1);
                let raw = match quantile_method {
                    QuantileMethod::AveragedInvertedCdf => percentiles_aicdf(&col, &levels),
                    QuantileMethod::Linear => percentiles_linear(&col, &levels),
                };
                // Deduplicate: keep edge if it's more than 1e-8 above its predecessor.
                // sklearn: np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                // The first edge always passes (to_begin=inf means infinite diff at start).
                let mut deduped: Vec<f64> = Vec::with_capacity(raw.len());
                for (i, &e) in raw.iter().enumerate() {
                    if i == 0 || e - deduped.last().copied().unwrap() > 1e-8 {
                        deduped.push(e);
                    }
                }
                deduped
            }
        };

        let effective_bins = edges.len() - 1;
        fitted_n_bins.push(effective_bins);
        bin_edges.push(edges);
    }

    Fitted {
        n_bins: fitted_n_bins,
        bin_edges,
    }
}

/// Transform one row in place: each `data[i*n_cols + j]` → bin index.
pub fn transform_row(row: &mut [f64], fitted: &Fitted) {
    for (j, v) in row.iter_mut().enumerate() {
        let edges = &fitted.bin_edges[j];
        // A collapsed column (single surviving edge, n_bins == 0) has an empty
        // inner-edge slice, so np.searchsorted maps every value to bin 0.
        if edges.len() < 2 {
            *v = 0.0;
            continue;
        }
        // Inner edges = edges[1..len-1] (strip outer fences).
        let inner = &edges[1..edges.len() - 1];
        // np.searchsorted(inner, v, side='right'): index where v would insert
        // to maintain sorted order from the right.
        let idx = inner.partition_point(|&e| e <= *v);
        // np.clip(idx, 0, n_bins-1)
        *v = idx.min(fitted.n_bins[j] - 1) as f64;
    }
}

/// Expand one ordinal-encoded row to one-hot-dense. The output length equals
/// `sum(n_bins)` for all columns.
pub fn onehot_row(ordinal_row: &[f64], fitted: &Fitted) -> Vec<f64> {
    let total_cols: usize = fitted.n_bins.iter().sum();
    let mut out = vec![0.0f64; total_cols];
    let mut offset = 0;
    for (j, &v) in ordinal_row.iter().enumerate() {
        let nb = fitted.n_bins[j];
        // A collapsed column contributes zero indicator columns; skip so its
        // bin-0 write does not land in the next column's block.
        if nb > 0 {
            out[offset + v as usize] = 1.0;
        }
        offset += nb;
    }
    out
}

/// One-hot column names derived from input column names.
pub fn onehot_col_names(col_names: &[String], n_bins: &[usize]) -> Vec<String> {
    col_names
        .iter()
        .zip(n_bins.iter())
        .flat_map(|(name, &nb)| (0..nb).map(move |k| format!("{name}_{k}")))
        .collect()
}

/// `np.linspace(start, stop, num)` — matches NumPy exactly.
///
/// NumPy computes `start + step * i` for each `i` in `0..num`, where
/// `step = (stop - start) / (num - 1)`, and then forces the last element
/// to equal `stop` exactly. This is the exact arithmetic used in the
/// `uniform` bin-edge computation.
pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    assert!(num >= 2);
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();
    *v.last_mut().unwrap() = stop;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(rows: &[&[f64]]) -> Vec<f64> {
        rows.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn linspace_matches_numpy() {
        // np.linspace(-2, 1, 4) = [-2, -1, 0, 1]
        let v = linspace(-2.0, 1.0, 4);
        assert_eq!(v, [-2.0, -1.0, 0.0, 1.0]);
        // endpoint is exact
        let v2 = linspace(0.1, 0.9, 9);
        assert_eq!(*v2.last().unwrap(), 0.9);
    }

    #[test]
    fn uniform_basic() {
        // sklearn example from docstring:
        // X = [[-2],[−1],[0],[1]], n_bins=3, strategy=uniform
        // col: min=-2, max=1, linspace(-2,1,4) = [-2,-1,0,1]
        let data = flat(&[&[-2.0], &[-1.0], &[0.0], &[1.0]]);
        let f = fit(
            &data,
            4,
            1,
            3,
            Strategy::Uniform,
            QuantileMethod::AveragedInvertedCdf,
        );
        assert_eq!(f.bin_edges[0], [-2.0, -1.0, 0.0, 1.0]);
        assert_eq!(f.n_bins[0], 3);

        // Transform: [-2,-1,0,1] → [0,1,2,2]
        let mut row = [-2.0f64];
        transform_row(&mut row, &f);
        assert_eq!(row[0], 0.0);

        let mut row = [1.0f64];
        transform_row(&mut row, &f);
        assert_eq!(row[0], 2.0);
    }

    #[test]
    fn constant_column_becomes_one_bin() {
        let data = flat(&[&[7.0], &[7.0], &[7.0]]);
        let f = fit(
            &data,
            3,
            1,
            5,
            Strategy::Quantile,
            QuantileMethod::AveragedInvertedCdf,
        );
        assert_eq!(f.n_bins[0], 1);
        assert_eq!(f.bin_edges[0], [f64::NEG_INFINITY, f64::INFINITY]);

        let mut row = [7.0f64];
        transform_row(&mut row, &f);
        assert_eq!(row[0], 0.0);
    }

    #[test]
    fn uniform_negative_values() {
        // col: [-10,-8,-6,-4,-2], n_bins=5
        // edges = linspace(-10,-2,6) = [-10,-8.4,-6.8,-5.2,-3.6,-2]
        let data: Vec<f64> = [-10.0f64, -8.0, -6.0, -4.0, -2.0].to_vec();
        let f = fit(
            &data,
            5,
            1,
            5,
            Strategy::Uniform,
            QuantileMethod::AveragedInvertedCdf,
        );
        // Each value should map to its index
        for (i, &v) in data.iter().enumerate() {
            let mut row = [v];
            transform_row(&mut row, &f);
            assert_eq!(row[0], i as f64, "i={i} data={v}");
        }
    }

    #[test]
    fn quantile_dedup_collapses_bins() {
        // Ties: col = [1,1,2,2,2,3], n_bins=5
        // For aicdf: levels=[0,20,40,60,80,100], n=6
        // h = 6*[0, 0.2, 0.4, 0.6, 0.8, 1.0] = [0, 1.2, 2.4, 3.6, 4.8, 6]
        // h=0: x[0]=1; h=1.2 (frac): x[1]=1; h=2.4 (frac): x[2]=2; h=3.6 (frac): x[3]=2; h=4.8(frac): x[4]=2; h=6: x[5]=3
        // raw = [1, 1, 2, 2, 2, 3] → after dedup (>1e-8): [1, 2, 3] → 2 bins
        let col: Vec<f64> = vec![1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
        let f = fit(
            &col,
            6,
            1,
            5,
            Strategy::Quantile,
            QuantileMethod::AveragedInvertedCdf,
        );
        assert_eq!(f.n_bins[0], 2);
        assert_eq!(f.bin_edges[0], [1.0, 2.0, 3.0]);
    }

    #[test]
    fn quantile_full_collapse_yields_zero_bins() {
        // Near-constant column: min != max but every quantile edge collapses
        // under the 1e-8 dedup, leaving a single edge. sklearn: n_bins_=0,
        // bin_edges_=[1.0], transform maps every value to 0 (no panic).
        let col = vec![1.0, 1.000000001, 1.0, 1.000000001];
        let f = fit(
            &col,
            4,
            1,
            5,
            Strategy::Quantile,
            QuantileMethod::AveragedInvertedCdf,
        );
        assert_eq!(f.n_bins[0], 0);
        assert_eq!(f.bin_edges[0], [1.0]);
        for &v in &col {
            let mut row = [v];
            transform_row(&mut row, &f);
            assert_eq!(row[0], 0.0);
        }
    }

    #[test]
    fn onehot_row_basic() {
        let f = Fitted {
            n_bins: vec![3, 2],
            bin_edges: vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 5.0, 10.0]],
        };
        // ordinal [1, 0] → onehot [0,1,0, 1,0]
        let row = [1.0f64, 0.0];
        let oh = onehot_row(&row, &f);
        assert_eq!(oh, [0.0, 1.0, 0.0, 1.0, 0.0]);
    }
}
