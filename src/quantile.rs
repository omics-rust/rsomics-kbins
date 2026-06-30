//! Percentile implementations matching sklearn 1.9.0's two quantile strategies.
//!
//! `sklearn.preprocessing.KBinsDiscretizer` offers two deterministic percentile
//! methods via `np.percentile`:
//!
//! - **`quantile_method="linear"` (NumPy type-7)**: virtual index
//!   `h = (n−1)·p`, linearly interpolates between the two bracketing order
//!   statistics. Used when `quantile_method="linear"`.
//!
//! - **`quantile_method="averaged_inverted_cdf"` (NumPy type-2, sklearn 1.9.0
//!   default)**: virtual index `h = n·p` (1-indexed). When `h` is an exact
//!   integer in (0, n), averages the two adjacent order statistics; otherwise
//!   takes `x[ceil(h)−1]` (0-indexed `x[floor(h)]`). The 0 and n boundary
//!   cases always return the min/max.

/// Type-7 (linear) percentile — `np.percentile(col, q, method="linear")`.
/// `q` is in percent [0, 100]; `xs` is reordered in place.
pub fn percentile_linear(xs: &mut [f64], q: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return xs[0];
    }
    let p = q / 100.0;
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 >= n {
        return select_kth(xs, n - 1, false).0;
    }
    let (a, b) = select_kth(xs, lo, true);
    a + frac * (b - a)
}

/// `averaged_inverted_cdf` percentile — `np.percentile(col, q, method="averaged_inverted_cdf")`.
/// This is NumPy type-2 (H&F 1996): `h = n·p`, averages adjacent order
/// statistics when `h` is an exact non-boundary integer.
/// `q` is in percent [0, 100]; `xs` is reordered in place.
pub fn percentile_aicdf(xs: &mut [f64], q: f64) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    let p = q / 100.0;
    let h = n as f64 * p;
    let j = h.floor() as usize; // 0-indexed floor
    let g = h - j as f64;

    if g == 0.0 {
        // Exact integer h
        if j == 0 {
            return select_kth(xs, 0, false).0;
        }
        if j >= n {
            return select_kth(xs, n - 1, false).0;
        }
        // Average of x[j-1] and x[j] (0-indexed)
        let (a, b) = select_kth(xs, j - 1, true);
        (a + b) * 0.5
    } else {
        // Fractional h: return x[j] (0-indexed = ceil(h)−1 in 1-indexed terms)
        if j >= n {
            return select_kth(xs, n - 1, false).0;
        }
        select_kth(xs, j, false).0
    }
}

/// The `k`-th order statistic (0-based) of `xs` by quickselect, plus the
/// next order statistic `k+1` when `want_next`. `xs` is reordered in place.
fn select_kth(xs: &mut [f64], k: usize, want_next: bool) -> (f64, f64) {
    let n = xs.len();
    let (_, &mut kth, right) = xs.select_nth_unstable_by(k, f64::total_cmp);
    if want_next && k + 1 < n {
        let next = right.iter().copied().fold(f64::INFINITY, f64::min);
        (kth, next)
    } else {
        (kth, kth)
    }
}

/// Multiple `averaged_inverted_cdf` percentiles over one column.
///
/// Sorts once (O(n log n)) then reads each level in O(1) via binary search on
/// the sorted copy — optimal for the k+1 evenly-spaced levels that
/// `KBinsDiscretizer` queries.
pub fn percentiles_aicdf(col: &[f64], levels_pct: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = col.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let n = sorted.len();

    levels_pct
        .iter()
        .map(|&q| {
            let p = q / 100.0;
            let h = n as f64 * p;
            let j = h.floor() as usize;
            let g = h - j as f64;

            if g == 0.0 {
                if j == 0 {
                    sorted[0]
                } else if j >= n {
                    sorted[n - 1]
                } else {
                    (sorted[j - 1] + sorted[j]) * 0.5
                }
            } else if j >= n {
                sorted[n - 1]
            } else {
                sorted[j]
            }
        })
        .collect()
}

/// Multiple type-7 (linear) percentiles over one column.
/// Sorts once then reads each level in O(1).
pub fn percentiles_linear(col: &[f64], levels_pct: &[f64]) -> Vec<f64> {
    let mut sorted: Vec<f64> = col.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
    let n = sorted.len();

    levels_pct
        .iter()
        .map(|&q| {
            let p = q / 100.0;
            let h = (n as f64 - 1.0) * p;
            let lo = h.floor() as usize;
            let frac = h - lo as f64;
            if lo + 1 >= n {
                return sorted[n - 1];
            }
            sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "got {a}, expected {b}");
    }

    fn bit_exact(a: f64, b: f64) {
        assert_eq!(a.to_bits(), b.to_bits(), "got {a}, expected {b}");
    }

    #[test]
    fn linear_type7_basic() {
        let c = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        close(percentile_linear(&mut c.clone(), 25.0), 2.0);
        close(percentile_linear(&mut c.clone(), 50.0), 3.0);
        close(percentile_linear(&mut c.clone(), 75.0), 4.0);
        close(percentile_linear(&mut c.clone(), 10.0), 1.4);
    }

    #[test]
    fn aicdf_basic() {
        // n=4, all boundaries are exact integers
        let c = [1.0f64, 2.0, 3.0, 4.0];
        close(percentile_aicdf(&mut c.clone(), 0.0), 1.0);
        close(percentile_aicdf(&mut c.clone(), 25.0), 1.5); // (x[0]+x[1])/2
        close(percentile_aicdf(&mut c.clone(), 50.0), 2.5); // (x[1]+x[2])/2
        close(percentile_aicdf(&mut c.clone(), 75.0), 3.5); // (x[2]+x[3])/2
        close(percentile_aicdf(&mut c.clone(), 100.0), 4.0);
    }

    #[test]
    fn aicdf_fractional_h() {
        // n=4, p=0.125: h=0.5, floor=0, g=0.5 (fractional) → x[0] = 1.0
        let c = [1.0f64, 2.0, 3.0, 4.0];
        close(percentile_aicdf(&mut c.clone(), 12.5), 1.0);
        // n=4, p=0.375: h=1.5, floor=1, g=0.5 (fractional) → x[1] = 2.0
        close(percentile_aicdf(&mut c.clone(), 37.5), 2.0);
    }

    #[test]
    fn percentiles_aicdf_matches_scalar() {
        let col = [3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let levels = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0];
        let batch = percentiles_aicdf(&col, &levels);
        for (i, &q) in levels.iter().enumerate() {
            let scalar = percentile_aicdf(&mut col.to_vec(), q);
            bit_exact(batch[i], scalar);
        }
    }

    #[test]
    fn percentiles_linear_matches_scalar() {
        let col = [3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let levels = [0.0, 25.0, 50.0, 75.0, 100.0];
        let batch = percentiles_linear(&col, &levels);
        for (i, &q) in levels.iter().enumerate() {
            let scalar = percentile_linear(&mut col.to_vec(), q);
            bit_exact(batch[i], scalar);
        }
    }

    #[test]
    fn sklearn_golden_aicdf_n5_quantile() {
        // sklearn 1.9.0: KBinsDiscretizer(n_bins=5, strategy='quantile')
        // default quantile_method='averaged_inverted_cdf'
        // col = [-0.85304393, 0.0660307, 0.1278404, 0.30471708, 0.94056472] (sorted)
        // percentile_levels = [0, 20, 40, 60, 80, 100] (linspace)
        // n=5, h = 5*[0, 0.2, 0.4, 0.6, 0.8, 1.0] = [0, 1, 2, 3, 4, 5]
        // All integers → averaging
        let col = [0.30471708f64, 0.94056472, 0.1278404, -0.85304393, 0.0660307];
        let levels = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0];
        let result = percentiles_aicdf(&col, &levels);
        // sorted: [-0.85304393, 0.0660307, 0.1278404, 0.30471708, 0.94056472]
        // h=0: sorted[0]
        // h=1: (sorted[0]+sorted[1])/2
        // h=2: (sorted[1]+sorted[2])/2
        // h=3: (sorted[2]+sorted[3])/2
        // h=4: (sorted[3]+sorted[4])/2
        // h=5: sorted[4]
        let sorted = [-0.85304393f64, 0.0660307, 0.1278404, 0.30471708, 0.94056472];
        close(result[0], sorted[0]);
        close(result[1], (sorted[0] + sorted[1]) * 0.5);
        close(result[2], (sorted[1] + sorted[2]) * 0.5);
        close(result[3], (sorted[2] + sorted[3]) * 0.5);
        close(result[4], (sorted[3] + sorted[4]) * 0.5);
        close(result[5], sorted[4]);
    }
}
