from __future__ import annotations

import pytest

from prob4d.cli import main


def test_grouped_cli_lists_provider_and_observation(capsys) -> None:
    assert main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "provider" in output
    assert "observation" in output
    assert "phystwin" in output
    assert "experiment" in output
    assert "identity" in output
    assert "prediction" in output
    assert "gauge" in output
    assert "installs only this grouped 'prob4d' executable" in output


def test_grouped_cli_lists_only_provider_v2_exports(capsys) -> None:
    assert main(["observation"]) == 0
    output = capsys.readouterr().out
    assert "export" in output
    assert "export-calibrated" in output
    assert "export-exploratory" in output
    assert "export-v1" not in output
    assert "bias-binding" in output
    assert "visual-bias" in output
    assert "visual-bias-stream" in output


def test_grouped_cli_requires_explicit_observation_export_mode(capsys) -> None:
    assert main(["observation", "export"]) == 2
    error = capsys.readouterr().err
    assert "intentionally ambiguous" in error
    assert "export-calibrated" in error
    assert "export-exploratory" in error
    assert "export-v1" not in error


def test_ambiguous_observation_export_help_is_informational(capsys) -> None:
    assert main(["observation", "export", "--help"]) == 0
    output = capsys.readouterr().out
    assert "Choose one explicit contract" in output
    assert "claim-bearing provider-v2 export" in output
    assert "labelled provider-v2 control" in output
    assert "prob4d-export-observation-belief" not in output


def test_grouped_cli_rejects_unknown_command(capsys) -> None:
    assert main(["unknown"]) == 2
    assert "usage: prob4d" in capsys.readouterr().err


def test_grouped_cli_lists_heldout_provider_experiment(capsys) -> None:
    assert main(["experiment"]) == 0
    output = capsys.readouterr().out
    assert "heldout-provider" in output


def test_grouped_cli_routes_promotion_and_identity_help(capsys) -> None:
    assert main(["experiment", "heldout-provider", "--help"]) == 0
    promotion_help = capsys.readouterr().out
    assert "seal a target-free promotion lock" in promotion_help

    assert main(["identity", "--help"]) == 0
    identity_help = capsys.readouterr().out
    assert "build-mixture" in identity_help
    assert "moment-match" in identity_help


def test_grouped_cli_routes_provider_neutral_prediction_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["prediction", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "import-motioncrafter" in output
    assert "import-vggt" in output
    assert "import-cut3r-online" in output
    assert "import-generic" in output
    assert "scaffold-generic" in output
    assert "validate" in output
    assert "runtime" in output


def test_grouped_cli_routes_generic_provider_import_help(capsys) -> None:
    with pytest.raises(SystemExit) as import_exit:
        main(["prediction", "import-generic", "--help"])
    assert import_exit.value.code == 0
    import_help = capsys.readouterr().out
    assert "external canonical predictions" in import_help
    assert "specification" in import_help

    with pytest.raises(SystemExit) as scaffold_exit:
        main(["prediction", "scaffold-generic", "--help"])
    assert scaffold_exit.value.code == 0
    scaffold_help = capsys.readouterr().out
    assert "no-clobber generic-provider import scaffold" in scaffold_help
    assert "output_directory" in scaffold_help


def test_grouped_cli_routes_provider_runtime_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["prediction", "runtime", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "causally selected provider-neutral" in output
    assert "inspect" in output
    assert "fuse-exploratory" in output


def test_grouped_cli_routes_vggt_import_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["prediction", "import-vggt", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "integrity-bound VGGT sample" in output
    assert "--sample-id" in output
    assert "--prediction-root" in output


def test_grouped_cli_routes_cut3r_online_import_help(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["prediction", "import-cut3r-online", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "recurrent-online CUT3R" in output
    assert "--cut3r-revision" in output
    assert "--checkpoint-sha256" in output
    assert "--confidence-threshold" in output


def test_grouped_cli_routes_common_mode_and_visual_bias_help(capsys) -> None:
    with pytest.raises(SystemExit) as stress_exit:
        main(["diagnostic", "common-mode-stress", "--help"])
    assert stress_exit.value.code == 0
    stress_help = capsys.readouterr().out
    assert "coherent visual-bias benchmark" in stress_help

    with pytest.raises(SystemExit) as bias_exit:
        main(["observation", "visual-bias", "--help"])
    assert bias_exit.value.code == 0
    bias_help = capsys.readouterr().out
    assert "explicit low-rank visual-bias nuisance" in bias_help


def test_grouped_cli_routes_recursive_visual_bias_help(capsys) -> None:
    with pytest.raises(SystemExit) as stream_exit:
        main(["observation", "visual-bias-stream", "--help"])
    assert stream_exit.value.code == 0
    stream_help = capsys.readouterr().out
    assert "recursive visual-bias nuisance streams" in stream_help


def test_grouped_cli_routes_observation_bias_binding_help(capsys) -> None:
    with pytest.raises(SystemExit) as binding_exit:
        main(["observation", "bias-binding", "--help"])
    assert binding_exit.value.code == 0
    binding_help = capsys.readouterr().out
    assert "Build, validate, or replay an exact" in binding_help
    assert "{build,validate,verify}" in binding_help


def test_grouped_cli_routes_sparse_gauge_prior_help(capsys) -> None:
    assert main(["gauge", "prior", "--help"]) == 0
    output = capsys.readouterr().out
    assert "verify" in output
    assert "materialize" in output
