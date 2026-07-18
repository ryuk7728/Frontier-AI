# CIFAR-10 Autoencoder Compression Experiment

This project trains the same autoencoder with different latent sizes and plots
the tradeoff between compression and reconstruction PSNR.

## Architecture

```text
3 x 32 x 32
-> 16 x 32 x 32
-> residual block
-> 64 x 16 x 16
-> residual block
-> 64 x 8 x 8
-> flatten 4096
-> latent
```

Both reductions in spatial resolution use learned stride-2 convolutions rather
than max pooling. The decoder is unchanged.

The decoder starts with `Linear(latent_size, 4096)`, reshapes to `64 x 8 x 8`,
and upsamples back to `3 x 32 x 32`.

The pre-latent and post-latent representations both contain 4096 values. This
means a latent size of 3072 no longer passes through the old hidden 1024-value
bottleneck. There are no encoder-to-decoder skip connections.

## Run

Download CIFAR-10 if needed:

```powershell
python dataset_prep.py
```

Edit `LATENT_SIZES`, `EPOCHS`, `LEARNING_RATE`, or `BATCH_SIZE` at the top of
`main.py`, then run:

```powershell
python main.py
```

The script writes:

- `results.json`
- `compression_vs_psnr.png`
- `models/latent_<size>.pt`
- `reconstructions/latent_<size>.png`

Compression is calculated as `3072 / latent_size`. Test PSNR is calculated from
the reconstruction MSE. The latent-512 target for this experiment is at least
27 dB.
