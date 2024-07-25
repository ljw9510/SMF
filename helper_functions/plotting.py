import re
import os
import numpy as np
import pandas as pd

from tqdm import trange
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image, ImageOps
#from pneumonia_dataprocess import process_path

#import seaborn as sns
#sns.set_theme()

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
})


def get_avg_stats(input_list, metric = "Accuracy"):
    method_xi_list = []
    for i in np.arange(len(input_list)):
        method = input_list[i].get("method")
        #method = method.replace(" (naive)", "")

        xi = input_list[i].get("xi")
        beta = input_list[i].get("beta")
        method_beta = [method, str(beta)]
        # print('_'.join([method, beta]))
        if [method_beta, xi] not in method_xi_list:
            method_xi_list.append([method_beta, xi])
        #if method not in method_xi_list:
        #    method_xi_list.append([method, xi])
    print(method_xi_list)

    results_list = []
    method_xi_list_new = []
    for method_xi in method_xi_list:
        method_beta, xi = method_xi[0], method_xi[1]
        avg_results = {}
        avg_results.update({"method":method_beta})
        avg_results.update({"xi":xi})

        avg_acc_list = []
        avg_rec_list = []
        for i in np.arange(len(input_list)):
            if (method_beta == [input_list[i].get("method"), str(input_list[i].get("beta"))]) and (xi == input_list[i].get("xi")):
                avg_acc_list.append(input_list[i].get(metric))
                print('!!! method, ACCU', [method_beta, input_list[i].get(metric)])
                avg_rec_list.append(input_list[i].get("Relative_reconstruction_loss (test)"))
                avg_results.update({"avg_acc_list": avg_acc_list})
                avg_results.update({"avg_rec_list": avg_rec_list})
        if method_xi not in method_xi_list_new:
            results_list.append(avg_results)
            method_xi_list_new.append(method_xi)

    return sorted(results_list, key=lambda d: d['method'])

def plot_accuracy(results_dict_list, save_path, metric="Accuracy", beta_list_plot=[1], ylim=[0,1], title=None):
    beta_list_plot = [str(beta) for beta in beta_list_plot]
    print('beta_list_plot', beta_list_plot)
    ncols = 1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7,5])

    avg_results_list = get_avg_stats(results_dict_list, metric=metric)

    color_dict = {}
    color_dict.update({'LR':'gray'})
    color_dict.update({'MF-LR':'k'})
    color_dict.update({'SDL-filt':'b'})
    color_dict.update({'SDL-feat':'r'})
    #color_dict.update({'SDL-feat (exhaustive)':'b'})
    color_dict.update({'SDL-conv-filt':'g'})
    color_dict.update({'SDL-conv-feat (naive)':'r'})
    #color_dict.update({'SDL-conv-feat (exhaustive)':'g'})
    marker_dict = {}
    marker_dict.update({'LR':'+'})
    marker_dict.update({'MR-->LR':">"})
    marker_dict.update({'SDL-filt':'*'})
    marker_dict.update({'SDL-feat':'x'})
    #marker_dict.update({'SDL-feat (exhaustive)':'^'})
    marker_dict.update({'SDL-conv-filt':''})
    marker_dict.update({'SDL-conv-feat (naive)':''})
    #marker_dict.update({'SDL-conv-feat (exhaustive)':''})

    # Get list of hyperparameters
    method_list = []
    xi_list = []

    for i in np.arange(len(avg_results_list)):
        result_dict = avg_results_list[i]
        xi = result_dict.get('xi')
        method = [ str(key) for key in result_dict.get('method')]
        if (xi is not None) and (xi not in xi_list):
            xi_list.append(xi)
        if method not in method_list:
            method_list.append(method)

    print("method_list",method_list)

    xi_list = sorted(xi_list)
    for method_beta in method_list:
        method, beta = method_beta[0], method_beta[1]
        print('method_beta', method_beta)

        if (method in ["LR", "MF-LR"]) or (beta in beta_list_plot):

            color = color_dict.get(method)
            marker = marker_dict.get(method)

            accuracy_array = []
            for j in np.arange(len(xi_list)):
                xi = xi_list[j]
                for i in np.arange(len(avg_results_list)):
                    results_dict = avg_results_list[i]
                    xi0 = results_dict.get('xi')
                    if (method == results_dict.get('method')[0]) and (beta == str(results_dict.get('method')[1])) and ((xi0 == xi) or (xi0==None)):
                        accuracy_array.append(results_dict.get('avg_acc_list').copy())
                        break

            accuracy_array = np.asarray(accuracy_array).T
            accuracy_mean = np.sum(accuracy_array, axis=0) / accuracy_array.shape[0]  ### axis-0 : trials
            accuracy_std = np.std(accuracy_array, axis=0)

            print("!!!accuracy_array", accuracy_array)

            if method == "LR":
                ax.axhline(y=accuracy_mean[0], color=color, linestyle='--', label="LR")
            else:
                linestyle = "-"
                if (method in ["LR", 'MF-LR']):
                    linestyle = "--"
                elif method in ["SDL-conv-filt", "SDL-conv-feat (naive)", "SDL-conv-feat (exhaustive)"]:
                    linestyle = "--"
                elif (beta != "None"):
                    print(method, beta)
                    #linestyle = "-"
                    s = method.split(" ")
                    if len(s) == 1:
                        method = s[0] #+ " (DR)"
                    else:
                        method = s[0] #+ " (DR) " + s[1]

                markers, caps, bars = ax.errorbar(xi_list, accuracy_mean, yerr=accuracy_std,
                                                       fmt=color, marker=marker, linestyle=linestyle, label=method, errorevery=20, markersize=10)
                ax.fill_between(xi_list, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, facecolor=color, alpha=0.1)


    #ax.title.set_text("[p, r, n, noise_std] = [%i, %i, %i, %.2f]" % (p,r,n, noise_std))
    ax.set_xlabel(r"$ \xi$", fontsize=12)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_ylim(ylim)
    ax.legend()
    if title is not None:
        plt.title(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(save_path)

def plot_pareto(results_dict_list, save_path,
                 metric="Accuracy", xlim=[0,1], ylim=[0,1],
                 beta_list_plot=[1],
                 title=None):
    "making pareto optimality plot.."

    ncols = 1
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[5,5])

    avg_results_list = get_avg_stats(results_dict_list, metric=metric)

    color_dict = {}
    color_dict.update({'LR':'gray'})
    color_dict.update({'MF-LR':'k'})
    color_dict.update({'SDL-filt':'b'})
    color_dict.update({'SDL-feat':'r'})
    #color_dict.update({'SDL-feat (exhaustive)':'b'})
    color_dict.update({'SDL-conv-filt':'g'})
    color_dict.update({'SDL-conv-feat':'y'})
    #color_dict.update({'SDL-conv-feat (exhaustive)':'g'})
    marker_dict = {}
    marker_dict.update({'LR':'+'})
    marker_dict.update({'MF-LR':">"})
    marker_dict.update({'SDL-filt':'*'})
    marker_dict.update({'SDL-feat':'x'})
    #marker_dict.update({'SDL-feat (exhaustive)':'^'})
    marker_dict.update({'SDL-conv-filt':'o'})
    marker_dict.update({'SDL-conv-feat (naive)':'p'})
    #marker_dict.update({'SDL-conv-feat (exhaustive)':'<'})

    xi_list = []
    for i in np.arange(len(avg_results_list)):
        result_dict = avg_results_list[i]
        xi = result_dict.get("xi")
        xi_list.append(xi)
    xi_list0 = list(set(xi_list))
    xi_list = [xi for xi in xi_list0 if xi is not None]
    xi_min = min(xi_list)
    print('xi_min', xi_min)

    for i in np.arange(len(avg_results_list)):
        result_dict = avg_results_list[i]
        method, beta = result_dict.get('method')[0], result_dict.get('method')[1]
        print(method, beta)
        xi = None
        if not ((method not in ["LR", "MF-LR"]) and (beta in beta_list_plot)):

            if method in ['SDL-filt', 'SDL-feat', 'SDL-conv-filt', 'SDL-conv-feat (naive)', 'SDL-conv-feat (exhaustive)']:
                xi = result_dict.get('xi')
            #print('xi', xi)

            rel_recons_error = np.mean(result_dict.get('avg_rec_list'))
            accuracy = np.mean(result_dict.get('avg_acc_list'))

            #print('rel_recons_error', rel_recons_error)
            #print('accuracy', accuracy)

            color = color_dict.get(method)
            marker = marker_dict.get(method)
            method0 = method
            if (beta != "None"):
                s = method.split(" ")

            print("method0", method0)
            print("rel_recons_error", rel_recons_error)
            print("accuracy", accuracy)

            if (xi is not None) and (xi>xi_min):
                ax.scatter(rel_recons_error, accuracy, s=100, c=color, alpha=1, marker=marker)
            else:
                ax.scatter(rel_recons_error, accuracy, s=100, c=color, alpha=1, label=method0.replace(" (naive)", ""), marker=marker)
            if (method in ['SDL-filt', 'SDL-conv-filt', 'SDL-conv-feat (naive)', 'SDL-feat']) and (xi in [0, 11]):
                x_len = xlim[1]-xlim[0]
                y_len = ylim[1]-ylim[0]
                ax.annotate(r" $\xi={}$".format(xi), (rel_recons_error-(0.07*x_len), accuracy+0.02*(y_len)), fontsize=9)

            results_prev = avg_results_list[i-1]
            if (xi is not None) and (xi>0) and (method == results_prev.get('method')[0]) and (beta == str(results_prev.get('method')[1]) ):
                #print(method, results_prev.get('method'), results_prev.get('xi'), xi)

                linestyle = "--"
                if (beta != "None"):
                    linestyle = "-"

                rel_recons_error_prev = np.mean(results_prev.get('avg_rec_list'))
                accuracy_prev = np.mean(results_prev.get('avg_acc_list'))
                line = [rel_recons_error_prev, accuracy_prev, rel_recons_error_prev, accuracy_prev]

                ax.plot([rel_recons_error_prev, rel_recons_error], [accuracy_prev, accuracy],
                        'k-', linestyle = linestyle, color = color)

            ax.set_xlabel('Reconstruction error', fontsize=12)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend(loc='lower right')
    if title is not None:
        plt.title(title, fontsize=13)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(save_path)


def plot_benchmark_errors(full_result_list, save_path,
                          method_list=None,
                          fig_size=[10,10],
                          xi_list_custom=None):
    if method_list is None:
        method_list = ["SDL-conv-feat (naive)", "SDL-conv-filt"]

    time_records = []
    errors = []
    f_interpolated_list = []

    methods_list = []
    stats_list = []
    xi_list = []
    for i in np.arange(len(full_result_list)):
        #if full_result_list[i].get("method")[0] not in ["LR", "MF-LR"]:
        method_name = full_result_list[i].get("method")[0]
        method_name = method_name.replace(" (naive)", "")
        if method_name in method_list:
            methods_list.append(full_result_list[i].get("method"))
            stats_list.append(np.asarray(full_result_list[i].get('avg_acc_list')))
            xi_list.append(full_result_list[i].get("xi"))
    print('methods_list', methods_list)
    #print('stats_list', stats_list)


    # max duration and time records
    x_all_max = 0
    for i in np.arange(len(stats_list)):
        errors0 = stats_list[i]
        x_all_max = max(x_all_max, max(errors0[:,0,-1]))

    x_all = np.linspace(0, x_all_max, num=101, endpoint=True)

    for i in np.arange(len(stats_list)):
        errors0 = stats_list[i] # trials x (time, error_data, error_label) x iterations
        time_records.append(x_all[x_all < min(errors0[:, 0, -1])])

    #print('time_records', len(time_records))

    # interpolate data and have common carrier

    for i in np.arange(len(stats_list)):
        errors0 = stats_list[i]
        f0_interpolated = []

        for j in np.arange(errors0.shape[0]): # trials for same setting
            f0 = interp1d(errors0[j, 0, :], xi_list[i]*errors0[j, 1, :]+errors0[j, 2, :], fill_value="extrapolate")
            x_all_0 = time_records[i]
            f0_interpolated.append(f0(x_all_0))
        f0_interpolated = np.asarray(f0_interpolated)
        f_interpolated_list.append(f0_interpolated)

    # make figure
    search_radius_const = full_result_list[0].get('search_radius_const')
    color_list = ['g', 'k', 'r', 'c', 'b']
    marker_list = ['*', '|', 'x', 'o', '+']
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    for i in np.arange(len(stats_list)):
        if xi_list_custom is None:
            xi_list_custom = xi_list

        if xi_list[i] in xi_list_custom:
            f0_interpolated = f_interpolated_list[i]
            f_avg0 = np.sum(f0_interpolated, axis=0) / f0_interpolated.shape[0]  ### axis-0 : trials
            f_std0 = np.std(f0_interpolated, axis=0)

            x_all_0 = time_records[i]
            color = color_list[i % len(color_list)]
            marker = marker_list[i % len(marker_list)]

            result_dict = full_result_list[i]
            beta = result_dict.get("beta")
            #if beta is None:
            #    label0 = result_dict.get("method")
            #else:
            #    # label0 = result_dict.get("method") + " ($\\beta=${}, $c'=${:.0f})".format(beta, search_radius_const)
            #    label0 = result_dict.get("method") + " ($\\beta=${}, $c'= \parallel X \parallel/10^5$)".format(beta)

            #print('methods_list[i]', methods_list[i])

            label0 = methods_list[i][0].replace(" (naive)", "") + " ($\\xi=${:.3f})".format(xi_list[i])

            markers, caps, bars = axs.errorbar(x_all_0, f_avg0, yerr=f_std0,
                                               fmt=color+'-', marker=marker, label=label0,
                                               markevery=5, markersize=10)
            axs.fill_between(x_all_0, f_avg0 - f_std0, f_avg0 + f_std0, facecolor=color, alpha=0.1)
            axs.set_ylim(ymax=10*np.max(f_avg0))
            axs.set_yscale('log')


    # min_max duration
    x_all_min_max = []
    for i in np.arange(len(time_records)):
        x_all_ALS0 = time_records[i]
        x_all_min_max.append(max(x_all_ALS0))

    x_all_min_max = min(x_all_min_max)
    axs.set_xlim(0, x_all_min_max)


    [bar.set_alpha(0.5) for bar in bars]
    # axs.set_ylim(0, np.maximum(np.max(f_OCPDL_avg + f_OCPDL_std), np.max(f_ALS_avg + f_ALS_std)) * 1.1)
    axs.set_xlabel('Elapsed time (s)', fontsize=15)
    axs.set_ylabel('Training loss', fontsize=15)
    data_name = full_result_list[0].get('data_name')
    title = data_name
    plt.suptitle(title, fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    axs.legend(fontsize=13, loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(0.15, 0.1, 0.9, 0.9, 0.00, 0.00)

    plt.savefig(save_path, bbox_inches='tight')
