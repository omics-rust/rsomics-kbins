# rsomics-kbins

Per-column binning — a value-exact, faster port of
`sklearn.preprocessing.KBinsDiscretizer` for the two deterministic strategies
(`uniform`, `quantile`).

```
cargo install rsomics-kbins
```

## Usage

```
rsomics-kbins --strategy uniform --n-bins 5 matrix.tsv
rsomics-kbins --strategy quantile --n-bins 10 matrix.tsv
rsomics-kbins --strategy uniform --n-bins 5 --encode onehot-dense matrix.tsv
rsomics-kbins --strategy quantile --n-bins 5 --json matrix.tsv
```

Input: a tab-separated `n×d` matrix (rows = samples, columns = features). A
leading empty top-left cell marks a header row; the first column then holds row
names. Without a header, rows and columns are numbered from 1.

Output (default): a TSV matrix of integer ordinal bin codes, with the same
row/column labels as the input. `--encode onehot-dense` expands each column
into its `n_bins` indicator columns.

`--json` emits a `{"row_names":…,"col_names":…,"matrix":[[…],…]}` envelope
(rsomics-common format).

### Options

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `quantile` | `uniform` or `quantile` |
| `--n-bins N` | `5` | Target bins per column (≥ 2) |
| `--encode` | `ordinal` | `ordinal` or `onehot-dense` |
| `--quantile-method` | `averaged-inverted-cdf` | `averaged-inverted-cdf` (sklearn 1.9+ default) or `linear` (type-7, sklearn <1.9 default) |
| `-t N` | — | Threads (via rsomics-common) |
| `--json` | — | JSON envelope output |

### `kmeans` strategy

The `kmeans` strategy is intentionally excluded. Its bin edges depend on
`KMeans` cluster centers seeded from a random state, making results
non-deterministic across runs and platforms. Uniform and quantile are the only
two fully deterministic strategies; value-exact bit-identity is achievable and
verified for both.

## Accuracy

Bin codes are integers — exact by construction. Bin edges use only arithmetic
and quantile interpolation (no transcendental functions), so they are
bit-identical to scikit-learn's `bin_edges_` on the same data.

- **`uniform`**: edges = `np.linspace(col_min, col_max, n_bins+1)` — pure
  FMA-equivalent arithmetic, bit-exact.
- **`quantile`**: edges = `np.percentile(col, linspace(0,100,n_bins+1),
  method=quantile_method)`. `averaged-inverted-cdf` matches sklearn 1.9.0's
  default (NumPy type-2); `linear` matches sklearn <1.9.0 (NumPy type-7).
  Consecutive-equal edges (≤ 1e-8 apart) are deduplicated exactly as sklearn
  does, collapsing bins for tied or constant columns.

Golden files in `tests/golden/` were generated once from sklearn 1.9.0
(numpy seed 42) and are frozen. The compat test runs the binary against them
without invoking Python.

## Origin

This crate is an independent Rust reimplementation of
`sklearn.preprocessing.KBinsDiscretizer` based on:

- The scikit-learn 1.9.0 source (`sklearn/preprocessing/_discretization.py`,
  BSD-3-Clause; reading and citing allowed — clean-room rule does not apply)
- The NumPy 2.4.6 percentile specification for `averaged_inverted_cdf`
  (type-2) and `linear` (type-7) methods

Golden test fixtures are generated from sklearn 1.9.0 with a fixed numpy seed.
The `quantile.rs` module independently implements the two interpolation
methods; no scikit-learn or NumPy source was copied verbatim.

License: MIT OR Apache-2.0.
Upstream credit: scikit-learn <https://scikit-learn.org> (BSD-3-Clause).
