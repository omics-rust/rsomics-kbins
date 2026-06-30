# Performance Notes — rsomics-kbins

## Machine

- **Host**: mini_m2 (Apple M2, macOS 25.5.0)
- **Ours version**: 0.1.0, `cargo build --release`
- **Upstream**: scikit-learn 1.9.0, NumPy 2.4.6, Python 3.12 (conda env `scanpy`)

## Fixture

- `/Volumes/KIOXIA/rsomics-fixtures/kbins/large_100k_20.tsv`
- Shape: 100 000 rows × 20 columns, standard-normal (numpy seed 42)
- File size: 40 MB

## Results

### Compute-only (fit + transform in-memory, single-thread)

Environment: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`

Tool | Strategy | Time (ms) | Speedup
-----|----------|-----------|-------
sklearn 1.9.0 | uniform n5 | 41 ms | 1.0×
**rsomics-kbins 0.1.0** | uniform n5 | **11.4 ms** | **3.6×**
sklearn 1.9.0 | quantile n5 | 110 ms | 1.0×
**rsomics-kbins 0.1.0** | quantile n5 | **50 ms** | **2.2×**

Ours measured via `cargo bench` (criterion, 100 samples).
sklearn measured via `time.perf_counter()`, 5 iterations.

### Both-serialize end-to-end (load TSV + fit_transform + write to /dev/null)

Tool | Strategy | Time (ms)
-----|----------|----------
sklearn 1.9.0 | uniform n5 | 807 ms
**rsomics-kbins 0.1.0** | uniform n5 | **255 ms** (3.2×)
sklearn 1.9.0 | quantile n5 | 815 ms
**rsomics-kbins 0.1.0** | quantile n5 | **257 ms** (3.2×)

Ours: `hyperfine --runs 10` (end-to-end binary including file I/O and TSV write).
sklearn: `time.perf_counter()`, includes `np.loadtxt` + `fit_transform` + `np.savetxt('/dev/null')`.

### Memory (peak RSS)

Tool | Strategy | RSS
-----|----------|----
sklearn 1.9.0 | uniform n5 | 171 MB
**rsomics-kbins 0.1.0** | uniform n5 | **95 MB** (1.8× less)

## Conclusion

Both compute-only and both-serialize axes exceed the 1.0× contract:

- Uniform: 3.6× compute, 3.2× end-to-end
- Quantile: 2.2× compute, 3.2× end-to-end
- RSS: 1.8× lower

The compute speedup for uniform comes from avoiding Python overhead and using
cache-friendly row-major scans. The quantile speedup comes from a single sort
per column followed by O(1) index lookups (vs sklearn's NumPy percentile which
allocates intermediate arrays).

The end-to-end speedup over sklearn is larger because `np.loadtxt` is slow
(Python parsing) while our TSV reader uses `fast-float2`.
