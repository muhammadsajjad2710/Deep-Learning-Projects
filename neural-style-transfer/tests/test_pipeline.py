from src.model import train


def test_training_returns_metrics():
    model, metrics = train()
    assert model['image'].shape[-1] == 16
    assert metrics['final_loss'] < metrics['start_loss']
