# Environment

The artifact is written for Python 3.10 or newer. The local validation pass used Python 3.12.

Install dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Required Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `plotnine`

The local validation pass used `numpy 2.4.4`, `pandas 3.0.1`, `matplotlib 3.10.8`, and `scikit-learn 1.8.0`.

The quick checks do not require a GPU, database server, cloud service, paid API, or private dataset. The full external-tabular suite uses plotnine's packaged example datasets in addition to scikit-learn datasets; the reviewer guide gives a scikit-learn-only command for shorter checks.

Figure regeneration writes PDF/PNG/SVG/HTML assets under `figures/`. The TikZ-based architecture figure requires a LaTeX installation with `pdflatex`; PNG export for that figure optionally uses `pdftoppm` when available. Full benchmark reruns can take longer than the test suite and should be run from a clean branch or separate output directory when preserving the committed evidence tables.
