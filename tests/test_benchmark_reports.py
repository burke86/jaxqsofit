import importlib.util
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(cv, rms, n=50):
    def bins(**extra):
        return [{"bin": b, "label": label, "n": n, "n_failed": 0, "median": 0.0, "nmad": 0.1, "rms": rms,
                 "outlier_fraction": 0.1, "coverage": 0.5, **extra}
                for b, label in enumerate(cv.LUMINOSITY_BIN_LABELS)]

    metrics = {name: bins() for name in cv.QUANTITIES}
    for kind in cv.LINE_PLOTS:
        metrics[f"line_{kind}"] = [m for _, base, _, _ in cv.LINES
                                   for m in bins(line=base, line_label=cv.LINE_NAMES[base])]
    return {**metrics, "elapsed_s": 600.0, "sha": "abc"}


def test_spectral_report_gates_bins_with_enough_objects():
    cv = _load("chimera_spectra_validation")
    limits = {str(b): {"max_outlier_fraction": 0.2, "max_rms": 0.5} for b in range(4)}
    rows = cv._report_rows(_metrics(cv, 0.3))
    thresholds = {key: limits for key, _, _ in rows}

    text, failures = cv.render_report(_metrics(cv, 0.4), thresholds, sha="abc", baseline=_metrics(cv, 0.3))
    assert failures == []
    assert "✅ passed" in text and "0.40 (+0.10) · 10%" in text

    _, failures = cv.render_report(_metrics(cv, 0.6), thresholds, sha="abc")
    assert len(failures) == len(rows) * 4

    # Too few objects in every bin: reported, not gated.
    text, failures = cv.render_report(_metrics(cv, 0.6, n=5), thresholds, sha="abc")
    assert failures == [] and "(N=5)" in text


def test_perf_report_fails_only_slowdowns_beyond_the_noise():
    bench = _load("sdss_pr_benchmark")

    def records(ms, compile_s=2.0):
        return [{"sha": "abcdef0", "ms_per_eval": m, "compile_s": compile_s, "final_loss": 1.0} for m in ms]

    base = records([1.00, 1.01, 0.99, 1.00, 1.02])
    _, failures = bench.render_report(base, records([1.30, 1.31, 1.29, 1.30, 1.32]))
    assert any("value + gradient" in f for f in failures)
    _, failures = bench.render_report(records([1.0, 0.6, 1.4, 1.0, 0.7]), records([1.2, 0.7, 1.7, 1.2, 0.8]))
    assert failures == []
    _, failures = bench.render_report(base, records([1.0] * 5, compile_s=3.0))
    assert any("compile" in f for f in failures)
