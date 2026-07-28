import pytest

from engine.bench.ledger import (
    LossEntry,
    LossLedger,
    MeasurementKind,
    Transformation,
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
