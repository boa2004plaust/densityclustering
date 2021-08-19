# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#
#!/bin/bash

BATCH=512
HEIGHT=256
WIDTH=256
POOL="gem"

# if HDBSCAN, set
CLUSTER="hdbscan"
EPS=5

MODEL="resnet_ibn50a"
DB="cub200"
MDIR="/home/deep/JiabaoWang/UnsupervisedFineGrain/DensityClustering/logs/${DB}_${CLUSTER}_${EPS}_resnet_ibn50a_gem_v2/checkpoint.pth.tar"
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/test.py -b ${BATCH} -a ${MODEL} -d ${DB} --cluster-method ${CLUSTER} --eps ${EPS} --resume ${MDIR} --height ${HEIGHT} --width ${WIDTH} --pooling-type ${POOL}

