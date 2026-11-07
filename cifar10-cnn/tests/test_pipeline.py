from src.model import train


def test_training_returns_metrics():
    model, metrics = train()
    assert model is not None
    assert metrics["accuracy"] >= 0.6
