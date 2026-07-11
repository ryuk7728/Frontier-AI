# CIFAR-10 Autoencoder Rate--Distortion Experiment

This project measures the tradeoff between latent compression and reconstruction
quality. A CIFAR-10 image contains `3 x 32 x 32 = 3072` values. For a latent of
size `z`, the reported compression ratio is `3072 / z` and reconstruction quality
is measured with whole-test-set PSNR.

## Architecture

For compressed settings (`z < 3072`), the encoder produces a `48 x 8 x 8 = 3072`
feature tensor and maps it to a spatial `z/64 x 8 x 8` latent tensor. The decoder
maps back from that latent. Every other internal activation contains at least 3072
values, so the configured latent is the only numerical bottleneck. There are no
encoder-to-decoder skip connections.

For `z = 3072`, `PixelUnshuffle(4)` and `PixelShuffle(4)` provide a reversible
path through a `48 x 8 x 8` latent. Identity-initialized 1x1 projections make the
no-compression endpoint exact; the CLI records it at epoch 0 without optimization.

## Verified result

The controlled 15-epoch sweep uses AdamW, OneCycleLR, learning rate `1e-3`, batch
size 128, seed 42, 45,000 training images, 5,000 validation images, and the fixed
10,000-image CIFAR-10 test set. The latent-512 model reached **30.36 dB test PSNR**,
exceeding the required 27 dB at 6x compression. The reversible 3072 model reached
79.83 dB (numerical near-identity).

## Run

```powershell
pip install -r requirements.txt
python dataset_prep.py
python main.py --mode train --latent-size 512
python main.py --mode sweep
```

Each run saves its best validation checkpoint, history, test metrics, and an
original/reconstruction image grid. Sweep mode also saves `sweep_results.json`
and `compression_vs_psnr.png`.

Evaluate a saved model with:

```powershell
python main.py --mode evaluate --checkpoint outputs/latent_512/best.pt
```

Only latent sizes divisible by 64 are supported because the latent is spatially
represented as `channels x 8 x 8`.
