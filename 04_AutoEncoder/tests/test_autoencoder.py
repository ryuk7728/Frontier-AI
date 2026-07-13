import torch
import torch.nn as nn

from src.model import AutoEncoder, ResidualBlock


def test_output_shape_matches_input():
    model = AutoEncoder(latent_size=512).eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert model(images).shape == images.shape


def test_prelatent_has_4096_values():
    model = AutoEncoder(latent_size=3072)
    encoder_linear = next(layer for layer in model.encoder if isinstance(layer, nn.Linear))
    decoder_linear = next(layer for layer in model.decoder if isinstance(layer, nn.Linear))
    assert encoder_linear.in_features == 4096
    assert encoder_linear.out_features == 3072
    assert decoder_linear.in_features == 3072
    assert decoder_linear.out_features == 4096


def test_encoder_uses_residual_blocks():
    model = AutoEncoder()
    residual_blocks = [layer for layer in model.encoder if isinstance(layer, ResidualBlock)]
    assert len(residual_blocks) == 2
