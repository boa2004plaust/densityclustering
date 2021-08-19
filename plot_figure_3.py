# encoding: utf-8
# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#

import os
import fitz
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


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


def embedding_imgs(x, y, img_px, img_py, acc_v, imgs, label_info, embedding_imgs_pth=None):
    fig1 = plt.figure(figsize=(12, 8))
    ax = fig1.add_subplot(111)
    ax.plot(x, y, 'b', label=label_info)

    box_offset = [(120, 165), (20, -200), (105, -180), (-120, -130)] #CUB
    # box_offset = [(140, -5), (20, 100), (105, 160), (-120, 180)]  # DOGS
    for i, (px, py, av, img) in enumerate(zip(img_px, img_py, acc_v, imgs)):
        ax.scatter(px, py, s=60)
        # ax.text(text_offset[i][0], text_offset[i][1], s="%d%%"%(av))
        img = Image.open(img)
        im = OffsetImage(img, zoom=0.2)
        im.image.axes = ax
        ab = AnnotationBbox(im, (px, py), xybox=box_offset[i], xycoords='data', boxcoords="offset points", pad=0,
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        # ax.annotate("%.1f" % (av), (px, py), xytext=text_offset[i],
        #             textcoords='offset points', arrowprops=dict(arrowstyle="->"))

    ax.axis([0, 50, 0, 60])
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Clustering ACC (%)', fontsize=15)
    ax.legend(fontsize=15, loc='lower right')
    # ax.grid(linestyle='-.')
    plt.tick_params(labelsize=15)
    # plt.axis('tight')
    plt.tight_layout()
    # plt.show()

    form = embedding_imgs_pth.split('.')[-1]
    plt.savefig(embedding_imgs_pth, format=form, dpi=300)


def pyMuPDF_fitz(pdfPath, imagePath):
    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        zoom_x = 1.33333333  # (1.33333333-->1152x864)
        zoom_y = 1.33333333
        mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pix = page.getPixmap(matrix=mat, alpha=False)

        if not os.path.exists(imagePath):
            os.makedirs(imagePath)
        pdfName = os.path.basename(pdfPath)
        pngName = pdfName.replace('.pdf', '.png')
        print(pngName)
        pix.writePNG(os.path.join(imagePath, pngName))


if __name__ == '__main__':
    label_info = 'CUB-200-2011'
    img_name = "Figure_3_%s.pdf" % label_info
    rootdir = '/home/deep/JiabaoWang/UnsupervisedFineGrain/DensityClustering/logs/'
    log_dir = os.path.join(rootdir, 'cub200_hdbscan_5_resnet_ibn50a_gem_v2')
    epoch, acc, nmi, ari = read_log_file_for_clustering(log_dir)
    # epoch, acc, nmi, ari, top1, top5 = read_log_file_for_classification(rootdir)

    x = [e for e in epoch]
    y = acc
    acc = acc
    img_px = [x[0], x[9], x[19], x[49]]
    img_py = [y[0], y[9], y[19], y[49]]
    acc_v = [acc[0], acc[9], acc[19], acc[49]]
    imgs = [os.path.join(log_dir, 'embedding_feat2d_train_points_0.pdf'),
            os.path.join(log_dir, 'embedding_feat2d_train_points_10.pdf'),
            os.path.join(log_dir, 'embedding_feat2d_train_points_20.pdf'),
            os.path.join(log_dir, 'embedding_feat2d_train_points_49.pdf'),]
    imagePath = log_dir
    for pdfPath in imgs:
        pyMuPDF_fitz(pdfPath, imagePath)
    imgs = [os.path.join(log_dir, 'embedding_feat2d_train_points_0.png'),
            os.path.join(log_dir, 'embedding_feat2d_train_points_10.png'),
            os.path.join(log_dir, 'embedding_feat2d_train_points_20.png'),
            os.path.join(log_dir, 'embedding_feat2d_train_points_49.png'), ]
    embedding_imgs(x, y, img_px, img_py, acc_v, imgs, label_info, embedding_imgs_pth=os.path.join(log_dir, img_name))
