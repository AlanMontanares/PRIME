import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

from matplotlib.cm import viridis, plasma, jet, ScalarMappable

#==================================================#
#==============CURVAS DE APRENDIZAJE===============#
#==================================================#


def show_curves(dataset =""):

    all_curves = [np.load(f"resultados/{dataset}/fold{i+1}\curvas.npy",allow_pickle=True).item() for i in range(5)]

    final_curve_means = {k: np.mean([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}
    final_curve_max = {k: np.max([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}
    final_curve_min = {k: np.min([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    #fig.set_facecolor('white')

    epochs = np.arange(len(final_curve_means["val_loss"])) + 1

    ax.plot(epochs, final_curve_means['val_loss'], label='Validation')
    ax.plot(epochs, final_curve_means['train_loss'], label='Training')
    ax.fill_between(epochs, y1=final_curve_min["val_loss"], y2= final_curve_max["val_loss"], alpha=.5)
    ax.fill_between(epochs, y1=final_curve_min["train_loss"], y2=final_curve_max["train_loss"], alpha=.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss evolution during training')
    ax.legend()

    print(f"Mejor Epoca Promedio: {final_curve_means['val_loss'].argmin()+1}, Loss: {final_curve_means['val_loss'].min()}")
    plt.show()


def show_all_curves(datasets = [], names= []):


    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    
    for x,dataset in enumerate(datasets):
        all_curves = [np.load(f"resultados/{dataset}/fold{i+1}\curvas.npy",allow_pickle=True).item() for i in range(2)]
        final_curve_means = {k: np.mean([c[k] for c in all_curves], axis=0) for k in all_curves[0].keys()}
        epochs = np.arange(len(final_curve_means["val_loss"])) + 1

        axs[0].plot(epochs, final_curve_means['train_loss'], label=f'{names[x]}')
        axs[1].plot(epochs, final_curve_means['val_loss'], label=f'{names[x]}')

        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Training Loss')
        axs[0].legend()

        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].set_title('Validation Loss')
        axs[1].legend()

    plt.show()

#==================================================#
#==============CURVAS DE APRENDIZAJE===============#
#==================================================#


#==================================================#
#=================PLOT REGRESION===================#
#==================================================#


def regression_metrics(y_true,y_pred):
    residuals = (y_pred- y_true)/(1+ y_true)

    bias = residuals.mean()
    nmad = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    foutliers = (np.abs(residuals)>0.05).sum()/len(residuals)
    return bias, nmad, foutliers



def plot_regresion(zphot1,zphot2, zspec1,zspec2,name1,name2,cmap = "jet", folds=5,dpi=700):

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi, sharey=True, gridspec_kw={'wspace': 0})


    font_size = 10
    vmax= 11


    metricas_1 = np.array([regression_metrics(
                                zspec1[i],
                                zphot1[i]) for i in range(folds)]).mean(0)

    metricas_2 = np.array([regression_metrics(
                                zspec2[i],
                                zphot2[i]) for i in range(folds)]).mean(0)

    #_, _, _, img  =axs[0].hist2d(zphot_1, zspec, bins=500, range=[[0, 0.32], [0, 0.32]], cmap='jet', cmin=1, vmax=vmax)
    #_, _, _, img2  =axs[1].hist2d(zphot_4, zspec, bins=500, range=[[0, 0.32], [0, 0.32]], cmap='jet', cmin=1, vmax=vmax)

    mean_frecuencias1 = []
    mean_frecuencias2 = []

    for i in range(folds):
        _, _, _, img  =axs[0].hist2d(zspec1[i], zphot1[i], bins=500, range=[[0, 0.32], [0, 0.32]], cmap=cmap,cmin=0.2, vmax=vmax)
        _, _, _, img2  =axs[1].hist2d(zspec2[i], zphot2[i], bins=500, range=[[0, 0.32], [0, 0.32]], cmap=cmap, cmin=0.2, vmax=vmax)

        mean_frecuencias1.append(img.get_array().data)
        mean_frecuencias2.append(img2.get_array().data)



    mean_frecuencias1 = np.nansum(np.array(mean_frecuencias1), axis=0)/5
    mean_frecuencias2 = np.nansum(np.array(mean_frecuencias2), axis=0)/5

    mean_frecuencias1[mean_frecuencias1==0] = np.nan
    mean_frecuencias2[mean_frecuencias2==0] = np.nan


    img.set_array(mean_frecuencias1)
    img2.set_array(mean_frecuencias2)

    axs[0].text(0.035, 0.3, f'{name1}', fontsize=12, color='black', ha='left', va='top')
    axs[0].text(0.035, 0.27, f"$<\\Delta z> =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[0].text(0.035, 0.25, f"$\\sigma_{{MAD}} =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[0].text(0.035, 0.23, f"$\\eta =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[0].text(0.09, 0.27, f"{metricas_1[0]:.5f}", fontsize=font_size, color='black')
    axs[0].text(0.076, 0.25, f"{metricas_1[1]:.5f}", fontsize=font_size, color='black')
    axs[0].text(0.055, 0.23, f"{metricas_1[2]*100:.2f}%", fontsize=font_size, color='black')


    axs[1].text(0.035, 0.3, f'{name2}', fontsize=12, color='black', ha='left', va='top')
    axs[1].text(0.035, 0.27, f"$<\\Delta z> =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[1].text(0.035, 0.25, f"$\\sigma_{{MAD}} =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[1].text(0.035, 0.23, f"$\\eta =$", fontsize=font_size, color='black', fontweight='bold', math_fontfamily='cm')
    axs[1].text(0.09, 0.27, f"{metricas_2[0]:.5f}", fontsize=font_size, color='black')
    axs[1].text(0.076, 0.25, f"{metricas_2[1]:.5f}", fontsize=font_size, color='black')
    axs[1].text(0.055, 0.23, f"{metricas_2[2]*100:.2f}%", fontsize=font_size, color='black')

    for ax in axs:

        ax.set_xlim(0, 0.32)
        ax.set_ylim(0, 0.32)
        ax.set_box_aspect(1)

        ax.tick_params(axis='both', which='major', length=6, bottom=True, top=True,left=True, right=True, direction='in')   
        ax.tick_params(axis='both', which='minor', length=3, left=True, right=True, bottom=True, top=True, direction='in')   

        ax.set_xticks(np.linspace(0.0,0.32,33), minor=True)
        ax.set_xticks(np.linspace(0.05,0.3,6), minor=False)

        ax.set_yticks(np.linspace(0.0,0.32,33), minor=True)
        ax.set_yticks(np.linspace(0.0,0.3,7), minor=False)

        ax.plot([0, 0.32], [0, 0.32], linestyle='-', color='black', linewidth=0.7)
        ax.plot([0.05, 0.32], [0, 0.257143], linestyle='--', color='red', alpha=0.7)
        ax.plot([0, 0.257143], [0.05, 0.32], linestyle='--', color='red', alpha=0.7)

        ax.set_xlabel("ZSPEC")


    axs[0].set_ylabel("ZPHOT")


    norm = Normalize(vmin=0.2, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', pad=0.03)
    cbar.set_label('DENSIDAD')

    plt.show()



#==================================================#
#=================PLOT REGRESION===================#
#==================================================#



#==================================================#
#====================PLOT BINS=====================#
#==================================================#


def metric_bin(zphot,zspec,feature,feature_name, metrica ="", survey="sdss"):
    res = ((zphot - zspec)/(1 + zspec))

    if feature_name == "z":
        intervalosz = np.arange(0, 0.31, 0.03)
    
    elif feature_name == "mag":
        intervalosz = np.arange(12, 17.8, 0.58)
    
    elif feature_name == "rad":
        intervalosz = np.geomspace(2, 100, 11)
    
    elif feature_name =="ebv":
        intervalosz = np.geomspace(0.002847, 0.18, 11)
    
    if survey =="sdss":
        if feature_name == "snr_u":
            intervalosz = np.geomspace(2, 180, 13)

        elif feature_name == "snr_g" or feature_name == "snr_r" or feature_name == "snr_i" or feature_name == "snr_z":
            intervalosz = np.geomspace(2, 650, 16)
    
    elif survey =="ps1":
        if feature_name == "snr_g" or feature_name == "snr_r" or feature_name == "snr_i" or feature_name == "snr_z":
            intervalosz = np.geomspace(50, 2000, 16)

    metricaz = np.zeros(len(intervalosz) - 1)

    for i in range(len(metricaz)):
        intervalo_inferior = intervalosz[i]
        intervalo_superior = intervalosz[i + 1]

        datos_intervalo = res[(feature >= intervalo_inferior) & (feature < intervalo_superior)]

        if metrica == "bias":
            metricaz[i] = (datos_intervalo).mean()

        elif metrica == "mad":
            metricaz[i] = 1.4826 * np.median(np.abs(datos_intervalo - np.median(datos_intervalo)))
        
        elif metrica == "outliers":
            metricaz[i] = (np.abs(datos_intervalo)>0.05).sum()/len(datos_intervalo)

        else:
            raise ValueError("Ingrese una métrica válida ['bias', 'mad', 'outliers']")
            
    return metricaz, intervalosz




def plot_metrics_bins(z_spect, zphots_list, feature, feature_name, names, alphas, fmts, capsizes, metric="bias",
                      cmap=plasma, ax=None, figsize=(7, 5), dpi=200, fontsize=14, fontsize_stick =12, ylim=0.03, survey="sdss"):
    
    colors = cmap(np.linspace(0.8, 0, len(zphots_list)))

    if ax is None:
        fig, axs = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        axs = ax
        fig = axs.figure

    ax3 = axs.twinx()
    
    feature_stack = np.hstack(feature)

    feature_min = 12 if feature_name == "mag" else (2 if feature_name == "rad" else 0.002847 if feature_name == "ebv" else None)
    feature_max = 17.8 if feature_name == "mag" else (100 if feature_name == "rad" else 0.18 if feature_name == "ebv" else None)

    if feature_name == "z":
        axs.hist(feature_stack, bins=np.linspace(0, 0.3, 150), color="gray", histtype="stepfilled", alpha=0.5, edgecolor="gray")
    else:
        bines = np.geomspace(feature_min, feature_max, 50)
        axs.hist(feature_stack, bins=bines, color="gray", histtype="stepfilled", alpha=0.5, edgecolor="gray")

    factorz = 0.015 

    for x in range(len(zphots_list)):
        promedios = []
        for i in range(len(zphots_list[x])):
            promedio, intervalosz = metric_bin(zphots_list[x][i], z_spect[i], feature=feature[i], feature_name=feature_name, metrica=metric, survey=survey)
            promedios.append(promedio)

        promedios = np.array(promedios)

        if feature_name == "z":
            ax3.errorbar(intervalosz[1:] - factorz, promedios.mean(axis=0), yerr=promedios.std(axis=0), fmt=fmts[x], 
                markeredgecolor='black', markersize=6, label=names[x], markeredgewidth=1, alpha=alphas[x], capsize=capsizes[x], color=colors[x])
        else: 
            med_x = (intervalosz[:-1] + intervalosz[1:]) / 2 
            ax3.errorbar(med_x, promedios.mean(axis=0), yerr=promedios.std(axis=0), fmt=fmts[x], 
                        markeredgecolor='black', markersize=6, label=names[x], markeredgewidth=1, alpha=alphas[x], capsize=capsizes[x], color=colors[x])

    if metric == "bias":
        ax3.yaxis.tick_left()
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_ticks_position('left')
        ax3.set_ylabel("$<\\Delta z>$", fontsize=fontsize, math_fontfamily='cm', va='bottom', ha='center')

        axs.yaxis.tick_right()
        axs.yaxis.set_label_position('right')
        axs.yaxis.set_ticks_position('right')

        if feature_name == "z":
            axs.text(0.311, 11000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$z$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0, 25000])

            if survey == "sdss":
                ax3.set_ylim([-0.013, 0.013])
                ax3.set_yticks(np.linspace(-0.01, 0.01, 5))

            elif survey == "ps1":
                ax3.set_ylim([-0.035, 0.035])
                ax3.set_yticks(np.linspace(-0.025, 0.025, 5))

            ax3.set_xlim([0, 0.3])

            ticks = np.linspace(0, 10000, 5)
            axs.set_yticks(ticks)

        elif feature_name == "mag":
            axs.text(18, 500000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$MAG$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0.1, 1000000000000])
            axs.set_yscale("log")
            
            minor_ticks = np.concatenate([
                np.arange(0.1, 1, 0.1),
                np.arange(1, 10, 1),
                np.arange(10, 100, 10),
                np.arange(100, 1000, 100),
                np.arange(1000, 10000, 1000),
                np.arange(10000, 100000, 10000)


            ])
            
            axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
            axs.set_yticks(minor_ticks, minor=True)
            axs.yaxis.set_minor_formatter(plt.NullFormatter())

            ax3.set_ylim([-0.01, 0.01])
            ax3.set_yticks(np.linspace(-0.0075, 0.0075, 5))
            ax3.set_xlim([12, 17.8])

        elif feature_name == "rad":
            axs.text(115, 500000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$PetroR90$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0.1, 1000000000000])
            axs.set_yscale("log")
            
            minor_ticks = np.concatenate([
                np.arange(0.1, 1, 0.1),
                np.arange(1, 10, 1),
                np.arange(10, 100, 10),
                np.arange(100, 1000, 100),
                np.arange(1000, 10000, 1000),
                np.arange(10000, 100000, 10000)


            ])
            
            axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
            axs.set_yticks(minor_ticks, minor=True)
            axs.yaxis.set_minor_formatter(plt.NullFormatter())

            ax3.set_ylim([-0.01, 0.01])
            ax3.set_yticks(np.linspace(-0.0075, 0.0075, 5))
            ax3.set_xlim([2, 100])
            ax3.set_xscale("log")

        elif feature_name == "ebv":
            axs.text(0.2, 500000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$E(B-V)$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0.1, 1000000000000])
            axs.set_yscale("log")
            
            minor_ticks = np.concatenate([
                np.arange(0.1, 1, 0.1),
                np.arange(1, 10, 1),
                np.arange(10, 100, 10),
                np.arange(100, 1000, 100),
                np.arange(1000, 10000, 1000),
                np.arange(10000, 100000, 10000)


            ])
            
            axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
            axs.set_yticks(minor_ticks, minor=True)
            axs.yaxis.set_minor_formatter(plt.NullFormatter())
            
            ax3.set_ylim([-0.01, 0.01])
            ax3.set_yticks(np.linspace(-0.0075, 0.0075, 5))
            ax3.set_xlim([0.002847, 0.18])
            ax3.set_xscale("log")

        ax3.axhline(y=0, color='gray', linestyle='dotted')

    elif metric == "mad":
        
        ax3.yaxis.tick_left()
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_ticks_position('left')
        ax3.set_ylabel("$\\sigma_{{MAD}}$", fontsize=fontsize, math_fontfamily='cm', va='bottom', ha='center')

        axs.yaxis.tick_right()
        axs.yaxis.set_label_position('right')
        axs.yaxis.set_ticks_position('right')

        if feature_name == "z":
            axs.text(0.311, 11000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$z$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0, 25000]) 
            ax3.set_xlim([0, 0.3])
    
            ticks = np.linspace(0, 10000, 5)
            axs.set_yticks(ticks)

        elif feature_name == "mag":
            axs.text(18, 500000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$MAG$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0.1, 1000000000000])
            axs.set_yscale("log")
            
            minor_ticks = np.concatenate([
                np.arange(0.1, 1, 0.1),
                np.arange(1, 10, 1),
                np.arange(10, 100, 10),
                np.arange(100, 1000, 100),
                np.arange(1000, 10000, 1000),
                np.arange(10000, 100000, 10000)


            ])
            
            axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
            axs.set_yticks(minor_ticks, minor=True)
            axs.yaxis.set_minor_formatter(plt.NullFormatter())

            ax3.set_xlim([12, 17.8])

        elif feature_name == "rad":
            axs.text(115, 500000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$PetroR90 \ ['']$", fontsize=18, math_fontfamily='cm')
            axs.set_ylim([0.1, 1000000000000])
            axs.set_yscale("log")
            
            minor_ticks = np.concatenate([
                np.arange(0.1, 1, 0.1),
                np.arange(1, 10, 1),
                np.arange(10, 100, 10),
                np.arange(100, 1000, 100),
                np.arange(1000, 10000, 1000),
                np.arange(10000, 100000, 10000)


            ])
            
            axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
            axs.set_yticks(minor_ticks, minor=True)
            axs.yaxis.set_minor_formatter(plt.NullFormatter())
            
            ax3.set_xlim([2, 100])
            ax3.set_xscale("log")
        
        elif feature_name == "ebv":
            axs.text(0.2, 500000, "N", fontsize=fontsize_stick, ha='left', va='bottom')
            axs.set_xlabel("$E(B-V)$", fontsize=fontsize, math_fontfamily='cm')
            axs.set_ylim([0.1, 1000000000000])
            axs.set_yscale("log")
            
            minor_ticks = np.concatenate([
                np.arange(0.1, 1, 0.1),
                np.arange(1, 10, 1),
                np.arange(10, 100, 10),
                np.arange(100, 1000, 100),
                np.arange(1000, 10000, 1000),
                np.arange(10000, 100000, 10000)


            ])
            
            axs.set_yticks([1, 10, 100, 1000,10000,100000], minor=False)    
            axs.set_yticks(minor_ticks, minor=True)
            axs.yaxis.set_minor_formatter(plt.NullFormatter())
            
            ax3.set_xlim([0.002847, 0.18])
            ax3.set_xscale("log")

        ax3.set_ylim([0, ylim])

    ax3.legend()

    # Ajustar el tamaño de las etiquetas de los ejes x e y
    axs.tick_params(axis='x', labelsize=fontsize_stick)
    axs.tick_params(axis='y', labelsize=fontsize_stick)
    ax3.tick_params(axis='y', labelsize=fontsize_stick)

    plt.tight_layout()

    if ax is None:
        plt.show()

    

def plot_snr_bins(z_spect, zphots,snrs,bands,fancy_bands,alphas, fmts, names, cmap, ylim, survey):
    
    colors = cmap(np.linspace(0.8, 0, len(zphots)))

    fig, axs = plt.subplots(len(bands), 2, figsize=(12, 3*len(bands)), sharex=True, gridspec_kw={'hspace': 0}, dpi=700)
    #plt.subplots_adjust(hspace=0)

    axes_bias = [axs[i,0].twinx() for i in range(len(bands))]
    axes_mad = [axs[i,1].twinx() for i in range(len(bands))]

    for i in range(len(snrs)):
        snr = np.hstack(snrs[i])
        bins = np.logspace(np.log10(min(snr)), np.log10(max(snr)), 250)
        axs[i, 0].hist(snr, bins=bins, color="gray", histtype="stepfilled", alpha=0.5, edgecolor="gray")
        axs[i, 1].hist(snr, bins=bins, color="gray", histtype="stepfilled", alpha=0.5, edgecolor="gray")

        for x in range(len(zphots)):

            promedios_bias = []
            promedios_mad = []
            for m in range(len(zphots[x])):
                promedio_bias, intervalos_bias = metric_bin(zphots[x][m],z_spect[m],feature=snrs[i][m],feature_name=f"snr_{bands[i]}", metrica ="bias", survey=survey)
                promedio_mad, intervalos_mad = metric_bin(zphots[x][m],z_spect[m],feature=snrs[i][m],feature_name=f"snr_{bands[i]}", metrica ="mad", survey=survey)

                promedios_bias.append(promedio_bias)
                promedios_mad.append(promedio_mad)

            promedios_bias = np.array(promedios_bias)
            promedios_mad = np.array(promedios_mad)

            med_bias = (intervalos_bias[:-1] + intervalos_bias[1:]) / 2
            med_mad = (intervalos_mad[:-1] + intervalos_mad[1:]) / 2

            axes_bias[i].errorbar(med_bias, promedios_bias.mean(axis=0), yerr=promedios_bias.std(axis=0), fmt=fmts[x],alpha =alphas[x],color=colors[x], markeredgecolor='black', markersize=6, label=names[x], markeredgewidth=0.7, linewidth=2)
            axes_mad[i].errorbar(med_mad, promedios_mad.mean(axis=0), yerr=promedios_mad.std(axis=0), fmt=fmts[x],alpha =alphas[x],color=colors[x], markeredgecolor='black', markersize=6, label=names[x], markeredgewidth=0.7, linewidth=2)

            axs[i,0].set_xscale("log")
            axs[i,1].set_xscale("log")


            #=========BIAS=========#
            axes_bias[i].yaxis.tick_left()
            axes_bias[i].yaxis.set_label_position('left')
            axes_bias[i].yaxis.set_ticks_position('left')
            axes_bias[i].set_ylabel("$<\\Delta z>$", fontsize=14, math_fontfamily='cm', va='bottom', ha='center')

            axes_bias[i].set_ylim([-0.006, 0.006])
            axes_bias[i].set_yticks(np.linspace(-0.004, 0.004, 5))

            #axes_bias[i].legend()
            axes_bias[i].axhline(y=0, color='gray', linestyle='dotted')
            #=========BIAS=========#

            #=========MAD=========#
            axes_mad[i].yaxis.tick_left()
            axes_mad[i].yaxis.set_label_position('left')
            axes_mad[i].yaxis.set_ticks_position('left')
            axes_mad[i].set_ylabel("$\\sigma_{{MAD}}$", fontsize=14, math_fontfamily='cm', va='bottom', ha='center')

            axes_mad[i].set_ylim([0, ylim])

            if ylim ==0.03:
                axes_mad[i].set_yticks(np.linspace(0.004, 0.028, 7))
            elif ylim == 0.015:
                axes_mad[i].set_yticks(np.linspace(0.002, 0.014, 7))

            #=========MAD=========#
            
            if i == 0:
                #axes_bias[i].legend()
                axes_mad[i].legend()

        axs[i,0].yaxis.set_ticks([])
        axs[i,1].yaxis.set_ticks([])


    for row in axs:
        for ax in row:
            if survey=="sdss":
                ax.set_xlim(2, 650)
            elif survey =="ps1":
                ax.set_xlim(50, 2000)
    
            

    for i, (ax1, ax2) in enumerate(zip(axs[:,0], axs[:,1])):
        ax1.text(0.025, 0.025, fancy_bands[i], transform=ax1.transAxes, fontsize=17, va='bottom', ha='left', math_fontfamily='cm')
        ax2.text(0.025, 0.025, fancy_bands[i], transform=ax2.transAxes, fontsize=17, va='bottom', ha='left', math_fontfamily='cm')


    axs[len(bands)-1,0].set_xlabel("$SNR$", fontsize=14, math_fontfamily='cm')
    axs[len(bands)-1,1].set_xlabel("$SNR$", fontsize=14, math_fontfamily='cm')
    
    plt.tight_layout()
    plt.show()

