# encoding: utf-8
# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#

import os
import matplotlib.pyplot as plt


def read_log_file_for_classification(log_path):
    _epoch, _acc, _nmi, _ari, _top1, _top5 = [], [], [], [], [], []
    with open(os.path.join(log_path, 'log_train.txt'), encoding='UTF-8') as f:
        line_list = f.readlines()
        for line in line_list:
            idx = line.find('#Classification Epoch ')
            if idx >= 0:
                idx += len('#Classification Epoch ')
                idx_end = line.find(', ACC: ')
                _epoch.append(int(line[idx:idx_end]))

                idx = line.find(', ACC: ')
                if idx >= 0:
                    idx += len(', ACC: ')
                    idx_end = line.find('%, NMI: ')
                    _acc.append(float(line[idx:idx_end]))

                idx = line.find(', NMI: ')
                if idx >= 0:
                    idx += len(', NMI: ')
                    idx_end = line.find('%, ARI: ')
                    _nmi.append(float(line[idx:idx_end]))

                idx = line.find(', ARI: ')
                if idx >= 0:
                    idx += len(', ARI: ')
                    idx_end = line.find('%, Top-1: ')
                    _ari.append(float(line[idx:idx_end]))

                idx = line.find(', Top-1: ')
                if idx >= 0:
                    idx += len(', Top-1: ')
                    idx_end = line.find('%, Top-5: ')
                    _top1.append(float(line[idx:idx_end]))

                idx = line.find(', Top-5: ')
                if idx >= 0:
                    idx += len(', Top-5: ')
                    idx_end = line.find('%.')
                    _top5.append(float(line[idx:idx_end]))

    return _epoch, _acc, _nmi, _ari, _top1, _top5


def read_log_file_for_clustering(log_path):
    _epoch, _acc, _nmi, _ari = [], [], [], []
    with open(os.path.join(log_path, 'log_train.txt'), encoding='UTF-8') as f:
        line_list = f.readlines()
        for line in line_list:
            idx = line.find('#Clustering Epoch ')
            if idx >= 0:
                idx += len('#Clustering Epoch ')
                idx_end = line.find(', ACC: ')
                _epoch.append(int(line[idx:idx_end]))

                idx = line.find(', ACC: ')
                if idx >= 0:
                    idx += len(', ACC: ')
                    idx_end = line.find('%, NMI: ')
                    _acc.append(float(line[idx:idx_end]))

                idx = line.find(', NMI: ')
                if idx >= 0:
                    idx += len(', NMI: ')
                    idx_end = line.find('%, ARI: ')
                    _nmi.append(float(line[idx:idx_end]))

                idx = line.find(', ARI: ')
                if idx >= 0:
                    idx += len(', ARI: ')
                    idx_end = line.find('%.')
                    _ari.append(float(line[idx:idx_end]))

    return _epoch, _acc, _nmi, _ari


def dual_axis_plot(top1, acces, curves_path):
    fig, ax1 = plt.subplots()
    legend1 = [
        'DeepCluster',
        'GroupSampling',
        'Ours',
        ]
    colors1 = [':b', '-.b', '-b']
    epochs = range(0, len(top1[0]))
    for i in range(len(top1)):
        p1, = ax1.plot(epochs, top1[i], colors1[i], label=legend1[i])

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Classification Top-1 (%)')
    ax1.set_xlim([0, len(epochs)])
    ax1.set_ylim([0, 60])

    ax2 = ax1.twinx()

    legend2 = [
        'DeepCluster',
        'GroupSampling',
        'Ours',
    ]
    colors2 = [':r', '-.r', '-r']
    epochs = range(0, len(acces[0]))
    for i in range(len(acces)):
        p2, = ax2.plot(epochs, acces[i], colors2[i], label=legend2[i])

    ax2.set_ylabel('Clustering ACC (%)')
    ax2.set_ylim([0, 60])

    ax1.yaxis.get_label().set_color(p1.get_color())
    ax2.yaxis.get_label().set_color(p2.get_color())
    ax1.tick_params(axis='y', colors=p1.get_color())
    ax2.tick_params(axis='y', colors=p2.get_color())

    leg1 = ax1.legend(loc='upper left')
    leg1.texts[0].set_color(p1.get_color())
    leg1.texts[1].set_color(p1.get_color())
    leg1.texts[2].set_color(p1.get_color())

    leg2 = ax2.legend(loc='upper right')
    leg2.texts[0].set_color(p2.get_color())
    leg2.texts[1].set_color(p2.get_color())
    leg2.texts[2].set_color(p2.get_color())

    ax1.grid(linestyle='-.')
    plt.tight_layout()
    form = curves_path.split('.')[-1]
    plt.savefig(curves_path, format=form, dpi=600)


if __name__ == '__main__':
    rootdir = '/home/deep/JiabaoWang/UnsupervisedFineGrain/DensityClustering/logs/'

    log_dir = os.path.join(rootdir, 'cub200_hdbscan_5_resnet_ibn50a_gem_v2')
    # _, nn, _, _, top1, top5 = read_log_file_for_classification(log_dir)
    # _, acc, _, _ = read_log_file_for_clustering(log_dir)
    # print(len(top1), top1)
    # print(len(acc), acc)
    deepcluster_top1 = [47.48, 41.94, 38.33, 34.21, 30.08, 28.77, 26.92, 25.23, 23.04, 22.61, 21.69, 21.56, 21.26, 21.4, 21.14, 20.99, 19.66, 20.11, 20.26, 20.62, 20.59, 18.76, 19.62, 19.83, 19.38, 19.42, 18.29, 19.02, 16.97, 18.67, 17.83, 17.47, 17.47, 17.64, 16.59, 18.67, 18.71, 17.55, 18.05, 16.91, 17.33, 16.17, 18.45, 18.64, 17.74, 17.05, 17.22, 17.16, 17.16, 16.74]
    deepcluster_acc = [26.74, 27.24, 27.53, 26.21, 25.73, 25.79, 25.71, 24.99, 24.32, 25.36, 23.97, 24.06, 23.81, 24.42, 23.79, 23.56, 23.47, 23.17, 23.84, 23.09, 23.76, 23.29, 23.24, 23.09, 22.17, 22.69, 22.89, 22.91, 22.42, 22.97, 22.37, 21.69, 22.17, 21.6, 22.49, 21.3, 22.12, 21.54, 22.52, 22.06, 21.34, 21.3, 21.91, 22.77, 22.61, 21.24, 21.96, 21.69, 21.81, 22.24]
    group_sampling_top1 = [26.3, 26.9, 21.6, 22.7, 22.5, 22.6, 20.5, 22.6, 22.0, 21.9, 16.2, 14.5, 20.4, 19.1, 15.4, 16.9, 21.0, 17.7, 14.8, 14.8, 15.6, 18.3, 19.9, 19.5, 18.3, 19.2, 15.8, 19.8, 16.9, 19.0, 18.8, 18.4, 18.3, 18.5, 18.0, 20.2, 16.8, 17.7, 17.0, 14.0, 13.8, 16.5, 18.3, 16.0, 15.9, 16.6, 15.1, 17.2, 18.1, 18.0]
    group_sampling_acc = [9.1, 5.7, 8.1, 7.5, 7.7, 8.6, 7.9, 8.4, 8.0, 7.8, 7.8, 8.2, 8.2, 8.7, 8.0, 8.8, 8.8, 8.1, 8.2, 7.5, 8.5, 8.8, 8.4, 8.5, 9.1, 8.7, 8.9, 9.1, 9.2, 8.8, 9.7, 9.1, 9.8, 9.5, 9.4, 9.9, 9.4, 9.4, 9.4, 9.3, 9.6, 9.9, 9.6, 9.3, 9.9, 9.5, 9.9, 9.1, 9.6, 10.2]
    densityclustering_top1 = [46.0, 27.9, 28.3, 31.2, 29.0, 34.1, 38.1, 39.2, 43.9, 42.6, 44.7, 45.9, 48.7, 47.8, 50.1, 48.3, 46.4, 51.0, 52.1, 50.7, 53.7, 55.8, 56.3, 57.6, 57.9, 58.7, 58.6, 58.1, 58.3, 58.4, 59.7, 59.5, 60.3, 59.6, 59.8, 59.3, 59.6, 59.3, 59.9, 59.8, 59.5, 59.3, 59.6, 60.0, 59.7, 59.5, 60.0, 60.0, 59.6, 59.8]
    densityclustering_acc = [15.4, 10.5, 11.7, 13.4, 13.0, 16.0, 19.0, 20.6, 23.6, 23.3, 24.9, 27.2, 29.2, 30.1, 31.3, 31.3, 30.6, 33.2, 33.8, 34.5, 35.3, 37.9, 38.9, 39.4, 39.9, 40.2, 40.5, 41.1, 41.3, 41.5, 42.1, 42.2, 42.6, 42.6, 43.0, 43.2, 43.3, 43.4, 43.6, 43.8, 43.9, 43.8, 43.8, 43.9, 44.0, 44.0, 43.8, 44.0, 44.0, 43.9]
    top1 = [deepcluster_top1, group_sampling_top1, densityclustering_top1]
    acces = [deepcluster_acc, group_sampling_acc, densityclustering_acc]
    dual_axis_plot(top1, acces, curves_path=os.path.join(log_dir, "Figure_1.pdf"))
