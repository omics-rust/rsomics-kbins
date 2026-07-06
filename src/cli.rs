use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use serde::Serialize;

use rsomics_common::{CommonFlags, Result, RsomicsError, ToolMeta, run};

use rsomics_kbins::{
    Fitted, Matrix, QuantileMethod, Strategy, fit, fmt_value, onehot_col_names, onehot_row,
    transform_row,
};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

/// Bin strategy matching `sklearn.preprocessing.KBinsDiscretizer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum StrategyArg {
    /// Equal-width bins per column.
    Uniform,
    /// Equal-frequency bins per column (quantile-based).
    Quantile,
}

/// Output encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EncodeArg {
    /// Integer bin index per cell (default).
    Ordinal,
    /// Expand each column into `n_bins` indicator columns.
    OnehotDense,
}

/// Per-column binning — value-exact `sklearn.preprocessing.KBinsDiscretizer` equivalent.
///
/// Reads a tab-separated `n×d` matrix (rows = samples, columns = features)
/// from a file argument or stdin (`-`). A leading empty top-left cell marks a
/// header row, and the first column holds row names. `--strategy` selects
/// `uniform` or `quantile`; `--n-bins` sets the target bin count per column.
///
/// Output is a TSV matrix of ordinal bin codes (default) or indicator columns
/// (`--encode onehot-dense`), with the same labeling convention as the input.
/// Pass `--json` for a `{"row_names":…,"col_names":…,"matrix":…}` envelope.
///
/// The `kmeans` strategy is excluded (stochastic; see README).
#[derive(Parser, Debug)]
#[command(name = "rsomics-kbins", version, about, long_about = None)]
pub struct Cli {
    /// Bin-edge strategy.
    #[arg(long = "strategy", value_enum, default_value = "quantile")]
    pub strategy: StrategyArg,

    /// Number of bins per column (≥ 2).
    #[arg(long = "n-bins", default_value_t = 5)]
    pub n_bins: usize,

    /// Output encoding: ordinal integer codes or one-hot indicator columns.
    #[arg(long = "encode", value_enum, default_value = "ordinal")]
    pub encode: EncodeArg,

    /// Quantile method for `--strategy quantile`.
    ///
    /// `averaged-inverted-cdf` matches sklearn 1.9.0 default; `linear` matches
    /// NumPy type-7 (sklearn <1.9.0 default).
    #[arg(
        long = "quantile-method",
        value_enum,
        default_value = "averaged-inverted-cdf"
    )]
    pub quantile_method: QuantileMethodArg,

    /// Feature matrix TSV (`-` or omitted reads stdin).
    #[arg(value_name = "MATRIX")]
    pub matrix: Option<PathBuf>,

    #[command(flatten)]
    pub common: CommonFlags,
}

/// Quantile method argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QuantileMethodArg {
    /// NumPy `averaged_inverted_cdf` (sklearn 1.9.0 default).
    AveragedInvertedCdf,
    /// NumPy `linear` / type-7.
    Linear,
}

#[derive(Serialize)]
struct MatrixOut {
    row_names: Vec<String>,
    col_names: Vec<String>,
    matrix: Vec<Vec<f64>>,
}

impl Cli {
    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            if self.n_bins < 2 {
                return Err(RsomicsError::InvalidInput(
                    "--n-bins must be at least 2".into(),
                ));
            }

            let m = Matrix::read(self.matrix.as_deref())?;
            if m.data.iter().any(|v| v.is_nan()) {
                return Err(RsomicsError::InvalidInput(
                    "input contains NaN/NA; KBinsDiscretizer does not accept missing values".into(),
                ));
            }

            let strategy = match self.strategy {
                StrategyArg::Uniform => Strategy::Uniform,
                StrategyArg::Quantile => Strategy::Quantile,
            };
            let qmethod = match self.quantile_method {
                QuantileMethodArg::AveragedInvertedCdf => QuantileMethod::AveragedInvertedCdf,
                QuantileMethodArg::Linear => QuantileMethod::Linear,
            };

            let fitted = fit(&m.data, m.n_rows, m.n_cols, self.n_bins, strategy, qmethod);

            let (data, col_names) = encode(&m, &fitted, self.encode)?;

            if !common.json {
                let stdout = std::io::stdout().lock();
                let mut w = BufWriter::new(stdout);
                write_tsv(&mut w, &m.row_names, &col_names, &data)?;
                w.flush().map_err(RsomicsError::Io)?;
            }

            let n_out_cols = col_names.len();
            Ok(MatrixOut {
                row_names: m.row_names,
                col_names,
                matrix: data.chunks(n_out_cols).map(<[f64]>::to_vec).collect(),
            })
        })
    }
}

fn encode(m: &Matrix, fitted: &Fitted, encode: EncodeArg) -> Result<(Vec<f64>, Vec<String>)> {
    match encode {
        EncodeArg::Ordinal => {
            let mut data = m.data.clone();
            for row in data.chunks_mut(m.n_cols) {
                transform_row(row, fitted);
            }
            Ok((data, m.col_names.clone()))
        }
        EncodeArg::OnehotDense => {
            let oh_col_names = onehot_col_names(&m.col_names, &fitted.n_bins);
            let n_out = oh_col_names.len();
            let mut data = Vec::with_capacity(m.n_rows * n_out);
            for i in 0..m.n_rows {
                let mut ordinal = m.row(i).to_vec();
                transform_row(&mut ordinal, fitted);
                data.extend_from_slice(&onehot_row(&ordinal, fitted));
            }
            Ok((data, oh_col_names))
        }
    }
}

fn write_tsv<W: Write>(
    w: &mut W,
    row_names: &[String],
    col_names: &[String],
    data: &[f64],
) -> Result<()> {
    writeln!(w, "\t{}", col_names.join("\t")).map_err(RsomicsError::Io)?;
    let n_cols = col_names.len();
    for (i, row) in data.chunks(n_cols).enumerate() {
        let cells: Vec<String> = row.iter().map(|&v| fmt_value(v)).collect();
        writeln!(w, "{}\t{}", row_names[i], cells.join("\t")).map_err(RsomicsError::Io)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
