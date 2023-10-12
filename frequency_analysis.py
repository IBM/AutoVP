from auto_vp.load_model import Load_Reprogramming_Model

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse


def To_Frequency(image):
    rgb_fft = torch.fft.fft2(image)
    rgb_fft = torch.fft.fftshift(rgb_fft, dim = (-2, -1))
    return rgb_fft


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset', choices=["CIFAR10", "CIFAR10-C", "CIFAR100", "Melanoma", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "UCF101", "FMoW"], required=True)
    args = p.parse_args()

    device = "cpu"

    # Load or build a reprogramming model
    reprogram_model = Load_Reprogramming_Model(args.dataset, device, file_path=f"{args.dataset}_last.pth") 

    # Draw the Prompts Result
    fig, ax = plt.subplots(1,2, figsize=(6,3))
    ax[0].imshow(np.transpose(np.float32(reprogram_model.input_perturbation.delta.cpu().detach().numpy()), (1,2,0)))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title("(a) The Learned Prompts", fontsize=10)
    fft_img = To_Frequency(reprogram_model.input_perturbation.delta)
    fft_img_mean = torch.mean(fft_img, dim=0)
    fft_img_mean = np.float32(torch.log(torch.abs(fft_img_mean)).detach().numpy())
    im = ax[1].imshow(fft_img_mean, cmap='gray', vmin=-1, vmax=8)
    ax[1].set_title("(b) Prompts in Frequency Domain", fontsize=10)
    ax[1].set_aspect('equal', adjustable='box')
    cbaxes = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, ax=ax[1], cax = cbaxes, ticks=[-1, 0, 4, 8])
    cbar.ax.set_yticklabels(['$e^{-1}$','$e^{0}$','$e^{4}$', '$e^{8}$'], fontsize=10)
    plt.savefig(f"image/{args.dataset}_prompt_fft") 

    