//! Criterion benchmarks: fit+transform vs sklearn KBinsDiscretizer.
//!
//! Fixture: `/Volumes/KIOXIA/rsomics-fixtures/kbins/large_100k_20.tsv`
//! (100 000 rows × 20 columns, standard-normal, generated from numpy seed 42).
//! Run: `cargo bench` — requires the fixture to be present (Tier-3).

use criterion::{Criterion, criterion_group, criterion_main};

use rsomics_kbins::{Matrix, QuantileMethod, Strategy, fit, transform_row};

fn bench_fit_transform_uniform(c: &mut Criterion) {
    let path = "/Volumes/KIOXIA/rsomics-fixtures/kbins/large_100k_20.tsv";
    if !std::path::Path::new(path).exists() {
        eprintln!("fixture not found, skipping bench: {path}");
        return;
    }
    let m = Matrix::read(Some(std::path::Path::new(path))).unwrap();
    c.bench_function("fit_transform_uniform_n5_100k_20", |b| {
        b.iter(|| {
            let fitted = fit(
                &m.data,
                m.n_rows,
                m.n_cols,
                5,
                Strategy::Uniform,
                QuantileMethod::AveragedInvertedCdf,
            );
            let mut data = m.data.clone();
            for row in data.chunks_mut(m.n_cols) {
                transform_row(row, &fitted);
            }
            std::hint::black_box(data)
        })
    });
}

fn bench_fit_transform_quantile(c: &mut Criterion) {
    let path = "/Volumes/KIOXIA/rsomics-fixtures/kbins/large_100k_20.tsv";
    if !std::path::Path::new(path).exists() {
        eprintln!("fixture not found, skipping bench: {path}");
        return;
    }
    let m = Matrix::read(Some(std::path::Path::new(path))).unwrap();
    c.bench_function("fit_transform_quantile_n5_100k_20", |b| {
        b.iter(|| {
            let fitted = fit(
                &m.data,
                m.n_rows,
                m.n_cols,
                5,
                Strategy::Quantile,
                QuantileMethod::AveragedInvertedCdf,
            );
            let mut data = m.data.clone();
            for row in data.chunks_mut(m.n_cols) {
                transform_row(row, &fitted);
            }
            std::hint::black_box(data)
        })
    });
}

criterion_group!(
    benches,
    bench_fit_transform_uniform,
    bench_fit_transform_quantile
);
criterion_main!(benches);
