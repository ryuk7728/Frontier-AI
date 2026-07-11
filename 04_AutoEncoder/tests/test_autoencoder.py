import torch

from src.model import AutoEncoder


def test_output_shape_matches_input():
    model = AutoEncoder(latent_size=512).eval()
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert model(images).shape == images.shape


def test_latent_is_the_only_internal_numerical_bottleneck():
    for latent_size in (64, 512, 3072):
        model = AutoEncoder(latent_size)
        assert model.pre_latent_values >= latent_size
        assert model.to_latent.out_channels * 8 * 8 == latent_size
        assert model.from_latent.in_channels * 8 * 8 == latent_size


def test_compression_ratio():
    assert AutoEncoder(512).compression_ratio == 6.0
    assert AutoEncoder(3072).compression_ratio == 1.0
