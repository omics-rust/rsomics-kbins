//! Value-exact compat tests against frozen sklearn 1.9.0 goldens.
//!
//! Golden inputs and expected outputs live in `tests/golden/` and were
//! generated once from `sklearn.preprocessing.KBinsDiscretizer` 1.9.0
//! (see `tests/golden/README`). Tests run the binary and compare output
//! byte-by-byte against the frozen expectation. Python is NOT invoked.

use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    // cargo test builds the binary into the same target dir
    let mut p = std::env::current_exe().unwrap();
    p.pop();
    if p.ends_with("deps") {
        p.pop();
    }
    p.push("rsomics-kbins");
    p
}

fn golden(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(name)
}

/// Run the binary and return trimmed stdout as a `Vec<Vec<i64>>` of bin codes
/// (strips the header row and the row-name first column).
fn run_ordinal(args: &[&str], input_file: &Path) -> Vec<Vec<i64>> {
    let out = Command::new(binary())
        .args(args)
        .arg(input_file)
        .output()
        .expect("binary not found");
    assert!(
        out.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    parse_codes(&stdout)
}

fn parse_codes(text: &str) -> Vec<Vec<i64>> {
    let mut lines = text.lines().peekable();
    // skip header if present (starts with tab)
    if lines.peek().map(|l| l.starts_with('\t')).unwrap_or(false) {
        lines.next();
    }
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut fields = line.split('\t');
            fields.next(); // row name
            fields
                .map(|v| v.trim().parse::<i64>().expect("non-integer code"))
                .collect()
        })
        .collect()
}

fn read_golden_codes(path: &Path) -> Vec<Vec<i64>> {
    let text = std::fs::read_to_string(path).expect("golden file missing");
    parse_codes(&text)
}

// ─── ordinal ──────────────────────────────────────────────────────────────────

#[test]
fn uniform_n3_ordinal() {
    let got = run_ordinal(
        &[
            "--strategy",
            "uniform",
            "--n-bins",
            "3",
            "--encode",
            "ordinal",
        ],
        &golden("basic_5x3.tsv"),
    );
    let expected = read_golden_codes(&golden("basic_5x3_uniform_n3_ordinal.tsv"));
    assert_eq!(got, expected, "uniform n3 ordinal codes must be bit-exact");
}

#[test]
fn quantile_n5_ordinal() {
    let got = run_ordinal(
        &[
            "--strategy",
            "quantile",
            "--n-bins",
            "5",
            "--encode",
            "ordinal",
        ],
        &golden("basic_5x3.tsv"),
    );
    let expected = read_golden_codes(&golden("basic_5x3_quantile_n5_ordinal.tsv"));
    assert_eq!(got, expected, "quantile n5 ordinal codes must be bit-exact");
}

#[test]
fn constant_col_quantile_n5_ordinal() {
    let got = run_ordinal(
        &[
            "--strategy",
            "quantile",
            "--n-bins",
            "5",
            "--encode",
            "ordinal",
        ],
        &golden("constant_col.tsv"),
    );
    let expected = read_golden_codes(&golden("constant_col_quantile_n5_ordinal.tsv"));
    assert_eq!(got, expected, "constant column collapse must match sklearn");
}

#[test]
fn negative_vals_uniform_n5_ordinal() {
    let got = run_ordinal(
        &[
            "--strategy",
            "uniform",
            "--n-bins",
            "5",
            "--encode",
            "ordinal",
        ],
        &golden("negative_vals.tsv"),
    );
    let expected = read_golden_codes(&golden("negative_vals_uniform_n5_ordinal.tsv"));
    assert_eq!(got, expected, "negative values uniform n5 ordinal");
}

#[test]
fn tied_col_quantile_n5_ordinal() {
    let got = run_ordinal(
        &[
            "--strategy",
            "quantile",
            "--n-bins",
            "5",
            "--encode",
            "ordinal",
        ],
        &golden("tied_col.tsv"),
    );
    let expected = read_golden_codes(&golden("tied_col_quantile_n5_ordinal.tsv"));
    assert_eq!(got, expected, "tied column dedup must match sklearn");
}

#[test]
fn normal_20x5_uniform_n5_ordinal() {
    let got = run_ordinal(
        &[
            "--strategy",
            "uniform",
            "--n-bins",
            "5",
            "--encode",
            "ordinal",
        ],
        &golden("normal_20x5.tsv"),
    );
    let expected = read_golden_codes(&golden("normal_20x5_uniform_n5_ordinal.tsv"));
    assert_eq!(got, expected, "20x5 uniform n5 ordinal");
}

#[test]
fn no_header_quantile_n10_ordinal() {
    // headerless input
    let got = run_ordinal(
        &[
            "--strategy",
            "quantile",
            "--n-bins",
            "10",
            "--encode",
            "ordinal",
        ],
        &golden("no_header_15x3.tsv"),
    );
    // golden is also headerless — parse without header-skipping
    let text = std::fs::read_to_string(golden("no_header_15x3_quantile_n10_ordinal.tsv")).unwrap();
    let expected: Vec<Vec<i64>> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            // no row names in headerless golden
            line.split('\t')
                .map(|v| v.trim().parse::<i64>().unwrap())
                .collect()
        })
        .collect();
    // got has auto-numbered row names, so strip them
    assert_eq!(got, expected, "headerless 15x3 quantile n10 ordinal");
}

// ─── onehot-dense ─────────────────────────────────────────────────────────────

fn run_onehot(args: &[&str], input_file: &Path) -> Vec<Vec<i64>> {
    let out = Command::new(binary())
        .args(args)
        .arg(input_file)
        .output()
        .expect("binary not found");
    assert!(
        out.status.success(),
        "binary failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    parse_codes(&stdout)
}

#[test]
fn uniform_n10_onehot_dense() {
    let got = run_onehot(
        &[
            "--strategy",
            "uniform",
            "--n-bins",
            "10",
            "--encode",
            "onehot-dense",
        ],
        &golden("medium_8x4.tsv"),
    );
    let expected = read_golden_codes(&golden("medium_8x4_uniform_n10_onehot.tsv"));
    assert_eq!(got, expected, "uniform n10 onehot-dense must be bit-exact");
}

#[test]
fn quantile_n3_onehot_dense() {
    let got = run_onehot(
        &[
            "--strategy",
            "quantile",
            "--n-bins",
            "3",
            "--encode",
            "onehot-dense",
        ],
        &golden("rand_10x3.tsv"),
    );
    let expected = read_golden_codes(&golden("rand_10x3_quantile_n3_onehot.tsv"));
    assert_eq!(got, expected, "quantile n3 onehot-dense must be bit-exact");
}
