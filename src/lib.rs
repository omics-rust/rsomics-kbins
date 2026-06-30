//! Per-column binning — a value-exact, faster port of
//! `sklearn.preprocessing.KBinsDiscretizer` for the two deterministic
//! strategies (`uniform`, `quantile`).
//!
//! Input is a numeric `n×d` matrix. Fit computes per-column bin edges;
//! transform replaces each value with its ordinal bin index. One-hot-dense
//! expansion flattens each column into its `n_bins` indicator columns.

mod io;
mod kbins;
mod quantile;

pub use io::{Matrix, fmt_value};
pub use kbins::{
    Fitted, QuantileMethod, Strategy, fit, linspace, onehot_col_names, onehot_row, transform_row,
};
pub use quantile::{percentile_aicdf, percentile_linear, percentiles_aicdf, percentiles_linear};
