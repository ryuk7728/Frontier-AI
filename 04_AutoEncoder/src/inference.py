import matplotlib.pyplot as plt


def plot_results(results):
    compression = [result["compression"] for result in results]
    psnr = [result["psnr"] for result in results]

    plt.plot(compression, psnr, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Compression ratio (3072 / latent size)")
    plt.ylabel("Test PSNR (dB)")
    plt.title("Compression vs reconstruction quality")
    plt.grid()
    plt.savefig("compression_vs_psnr.png")
    plt.close()
