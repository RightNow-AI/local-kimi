import pytest

from engine.bench.ledger import (
    LossEntry,
    LossLedger,
    MeasurementKind,
    Transformation,
    comparison_loss_ledger,
    standard_loss_ledger,
)


def test_unmeasured_entry_renders_as_unmeasured_and_never_zero():
    rendered = standard_loss_ledger().render()
    first_data_row = rendered.splitlines()[2]
    assert "UNMEASURED" in first_data_row
    assert " | 0 | " not in first_data_row


def test_total_loss_refuses_unmeasured_contributor():
    with pytest.raises(ValueError, match="UNMEASURED"):
        standard_loss_ledger().total_loss("quality_loss")


def test_total_loss_preserves_modelled_status_and_arithmetic():
    ledger = LossLedger(
        [
            LossEntry(
                Transformation.KERNEL_NUMERICS,
                "Replace one kernel",
                "delta_nll",
                "HuggingFace logits",
                MeasurementKind.MEASURED,
                0.02,
                "0.06 nats / 3 tokens = 0.02 nats/token",
            ),
            LossEntry(
                Transformation.SPECULATIVE_DECODING,
                "Model an acceptance loss",
                "delta_nll",
                "Direct engine decoding",
                MeasurementKind.MODELLED,
                0.01,
                "0.02 projected nats / 2 token classes = 0.01 nats/token",
            ),
        ]
    )
    total = ledger.total_loss("delta_nll")
    assert total.value == pytest.approx(0.03)
    assert total.kind is MeasurementKind.MODELLED
    assert total.arithmetic == "0.02 + 0.01 = 0.03"


def test_numeric_entry_requires_arithmetic():
    with pytest.raises(ValueError, match="arithmetic"):
        LossEntry(
            Transformation.REDUCED_TOP_K,
            "Reduce routed experts",
            "delta_nll",
            "Published top-k",
            MeasurementKind.MEASURED,
            0.1,
        )


def test_comparison_ledger_marks_only_observed_metrics_measured():
    ledger = comparison_loss_ledger(
        {
            "mean_token_kl_nats": 0.001,
            "mean_token_kl_arithmetic": "0.002 / 2 = 0.001",
            "top1_agreement": 0.5,
            "top1_arithmetic": "1 / 2 = 0.5",
            "routing_agreement": 0.75,
            "routing_arithmetic": "3 / 4 = 0.75",
            "perplexity_relative_delta": 0.02,
            "perplexity_arithmetic": "(5.1 - 5.0) / 5.0 = 0.02",
        },
        candidate_label="engine candidate",
        reference="HuggingFace checkpoint",
    )

    measured = [entry for entry in ledger.entries if entry.kind is MeasurementKind.MEASURED]
    assert {entry.metric for entry in measured} == {
        "mean_token_kl_nats",
        "top1_agreement",
        "routing_agreement",
        "perplexity_relative_delta",
    }
    with pytest.raises(ValueError, match="UNMEASURED"):
        ledger.total_loss("quality_loss")
