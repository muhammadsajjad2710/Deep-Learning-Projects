from src.model import train


def test_training_returns_metrics():
    model, metrics = train()
    assert model['generator'] is not None
    assert metrics['g_loss'] == metrics['g_loss']
    assert metrics['d_loss'] == metrics['d_loss']
