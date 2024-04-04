import numpy as np
import matplotlib.pyplot as plt

import os
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

###--------------METRICAS----------------###

def show_curves(curves, ruta, save=False):
    
    f, axs = plt.subplots(1, 1, figsize=(8, 5))

    epoch = np.arange(1, len(curves["train_loss"])+1)
    
    axs.plot(epoch, torch.tensor(curves["train_loss"]), label = "Train Loss")
    axs.plot(epoch, curves["val_loss"], label = "Val Loss")

    axs.set_title("Evolution of Loss during training")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    axs.legend()
    
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(ruta, "curves.png"), bbox_inches='tight')
        np.save(os.path.join(ruta, "curvas"), curves)

    #plt.show()


def regression_metrics(y_true,y_pred):
    residuals = (y_pred- y_true)/(1+ y_true)

    bias = residuals.mean()
    nmad = 1.4826 * torch.median((residuals - torch.median(residuals)).abs())
    foutliers = (residuals.abs()>0.05).sum()/len(residuals)
    return bias, nmad, foutliers



def plot_regression(z_spec, zphot, ruta, gridsize=200, scatter=True, save=False, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    ax_x_top = ax.twiny()
    ax_y_right = ax.twinx()

    ax.plot([0, 0.32], [0, 0.32], color='gray', linestyle='--')
    if scatter:
        xy = np.stack([z_spec.numpy(), zphot.numpy()])
        z = gaussian_kde(xy)(xy)

        idx = z.argsort()
        x, y, z = z_spec[idx], zphot[idx], z[idx]

        sc = ax.scatter(x, y, c=z, s=1, cmap="plasma") #ORIGINALMENTE S=2
        cbar = plt.colorbar(sc, ax=ax, label='Galaxy Density')

    else:
        hexplot = ax.hexbin(z_spec, zphot, gridsize=gridsize, cmap='plasma', bins="log")
        cbar = plt.colorbar(hexplot, ax=ax, label="Galaxy Density")

    font_size = 10

    ax.set_ylabel("ZPHOT",fontsize=9.5) 
    ax.set_xlabel("ZSPEC",fontsize=9.5) 

    ax.set_xlim([0, 0.32])
    ax.set_ylim([0, 0.32])

    ax_x_top.set_xlim([0, 0.32])
    ax_y_right.set_ylim([0, 0.32])

    ax_x_top.get_xaxis().set_ticklabels([])
    ax_y_right.get_yaxis().set_ticklabels([])

    division_length = 5 
    ax.tick_params(axis='both', direction='in', length=division_length, labelsize=font_size, size=4)
    ax_x_top.tick_params(direction='in', length=division_length, size=4)
    ax_y_right.tick_params(axis='both', direction='in', length=division_length, size=4)

    
    xticklabels = [0.05,0.10,0.15,0.20,0.25,0.30]
    ax.set_xticks(xticklabels)
    ax.set_xticklabels([f"{xtick:.2f}" for xtick in xticklabels])

    bias, nmad, foutliers = regression_metrics(z_spec,zphot)

    ax.text(0.03, 0.27, f"$<\\Delta z> =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    ax.text(0.03, 0.252, f"$\\sigma_{{MAD}} =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    ax.text(0.03, 0.234, f"$\\eta =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')

    ax.text(0.088, 0.27, f"{-bias:.5f}", fontsize=font_size, color='black')
    ax.text(0.073, 0.252, f"{nmad:.5f}", fontsize=font_size, color='black')
    ax.text(0.049, 0.234, f"{foutliers*100:.2f}%", fontsize=font_size, color='black')
    
    #plt.tight_layout()

    if save:
        plt.savefig(os.path.join(ruta, "regression.png"), bbox_inches='tight')

    #plt.show()


def plot_bias(zphot,zspec,ebv,rad,ruta,save=False):

    fig, axs = plt.subplots(1,4, figsize=(24,5), dpi=200)

    #============ BIAS ===========#
    res = ((zphot - zspec)/(1 + zspec))

    axs[0].hist(res, bins="auto",histtype = "step")
    axs[0].set_xlim([-0.15,0.15])
    axs[0].axvline(x=0, color='gray', linestyle='--')
    axs[0].set_ylabel("# of samples")
    axs[0].set_xlabel("$\\Delta z$")

    #============ BIAS ===========#

    #============ EBV ===========#
    intervalos = np.arange(0, 0.181, 0.02)

    promedios = np.zeros(len(intervalos) - 1)
    desviaciones_estandar = np.zeros(len(intervalos) - 1)

    for i in range(len(promedios)):
        intervalo_inferior = intervalos[i]
        intervalo_superior = intervalos[i + 1]

        datos_intervalo = res[(ebv >= intervalo_inferior) & (ebv < intervalo_superior)].numpy()

        promedios[i] = np.mean(datos_intervalo)
        desviaciones_estandar[i] = np.std(datos_intervalo)


    axs[1].hist(ebv, bins="auto", color="gray",histtype="stepfilled", alpha=0.5, edgecolor="gray")

    axs[1].set_yticks([])
    axs[1].set_yticklabels([])

    ax2 = axs[1].twinx()

    ax2.errorbar(intervalos[1:] - 0.01, promedios, yerr=desviaciones_estandar, fmt="s", color='r', markeredgecolor='black', markersize=6)

    ax2.yaxis.tick_left()

    ax2.set_ylabel("$\\Delta z$", ha='right')
    axs[1].set_xlabel("E(B-V)")

    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')
    ax2.set_ylim([-0.05,0.05])
    ax2.set_xlim([0,0.18])

    ax2.axhline(y=0, color='gray', linestyle='dotted')
    #============ EBV ===========#

    #============ z ===========#
    intervalosz = np.arange(0, 0.31, 0.012)

    promediosz = np.zeros(len(intervalosz) - 1)
    desviaciones_estandarz = np.zeros(len(intervalosz) - 1)

    for i in range(len(promediosz)):
        intervalo_inferior = intervalosz[i]
        intervalo_superior = intervalosz[i + 1]

        datos_intervalo = res[(zspec >= intervalo_inferior) & (zspec < intervalo_superior)].numpy()

        promediosz[i] = np.mean(datos_intervalo)
        desviaciones_estandarz[i] = np.std(datos_intervalo)


    axs[2].hist(zspec, bins="auto", color="gray",histtype="stepfilled", alpha=0.5, edgecolor="gray")

    axs[2].set_yticks([])
    axs[2].set_yticklabels([])

    ax3 = axs[2].twinx()

    ax3.errorbar(intervalosz[1:] - 0.006, promediosz, yerr=desviaciones_estandarz, fmt="s", color='r', markeredgecolor='black', markersize=6)

    ax3.yaxis.tick_left()

    ax3.set_ylabel("$\\Delta z$", ha='right')
    axs[2].set_xlabel("z")

    ax3.yaxis.set_label_position('left')
    ax3.yaxis.set_ticks_position('left')
    ax3.set_ylim([-0.05,0.05])
    ax3.set_xlim([0,0.3])

    ax3.axhline(y=0, color='gray', linestyle='dotted')
    #============ z ===========#

    #============ rad ===========#
    intervalosrad = np.arange(0, 51, 2.5)

    promediosrad = np.zeros(len(intervalosrad) - 1)
    desviaciones_estandarrad = np.zeros(len(intervalosrad) - 1)

    for i in range(len(promediosrad)):
        intervalo_inferior = intervalosrad[i]
        intervalo_superior = intervalosrad[i + 1]

        datos_intervalo = res[(rad >= intervalo_inferior) & (rad < intervalo_superior)].numpy()

        promediosrad[i] = np.mean(datos_intervalo)
        desviaciones_estandarrad[i] = np.std(datos_intervalo)


    axs[3].hist(rad, bins="auto", color="gray",histtype="stepfilled", alpha=0.5, edgecolor="gray")

    axs[3].set_yticks([])
    axs[3].set_yticklabels([])

    ax4 = axs[3].twinx()

    ax4.errorbar(intervalosrad[1:] - 1.25, promediosrad, yerr=desviaciones_estandarrad, fmt="s", color='r', markeredgecolor='black', markersize=6)

    ax4.yaxis.tick_left()

    ax4.set_ylabel("$\\Delta z$", ha='right')
    axs[3].set_xlabel("PetroR90")

    ax4.yaxis.set_label_position('left')
    ax4.yaxis.set_ticks_position('left')
    ax4.set_ylim([-0.05,0.05])
    ax4.set_xlim([0,50])

    ax4.axhline(y=0, color='gray', linestyle='dotted')
    #============ z ===========#
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(ruta, "bias.png"), bbox_inches='tight')

    #plt.show()


def plot_results(model, data, zphot,ruta,save=True):
    


    zspec = torch.tensor(data["metadata"].item()["z"].values)
    show_curves(model.curves, ruta=ruta, save=save)

    plot_regression(zspec,
                    zphot,
                    ruta=ruta, 
                    gridsize=200,
                    scatter=True,
                    save=save,
                    figsize=(6,4.8),
                    dpi=300)
    
    ebv = torch.tensor(data["metadata"].item()["EBV"].values).float()
    rad = torch.tensor(data["metadata"].item()["petroR90_r"].values).float()
    
    plot_bias(zphot,zspec,ebv,rad,ruta=ruta, save=save)