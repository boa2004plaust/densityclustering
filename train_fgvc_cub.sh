# Copyright (c) 2020-present.
# All rights reserved.
#
# Date: 2021-07-14
# Author: Jiabao Wang
# Email: jiabao_1108@163.com
#
#!/bin/bash

MOM=0.1
BATCH=192
HEIGHT=256
WIDTH=256
ITER=100
EPOCHES=50
Inst=16
Eval=1
POOL="gem"


### if HDBSCAN, set
CLUSTER="hdbscan"
EPS=5

### if DBSCAN, set
#CLUSTER="dbscan"
#EPS=0.4

MODEL="resnet_ibn50a"
DB="cub200"
LOG="logs/${DB}_${CLUSTER}_${EPS}_${MODEL}_${POOL}_v2"
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/train.py -b ${BATCH} -a ${MODEL} -d ${DB} --logs-dir ${LOG} --momentum ${MOM} --eps ${EPS} --height ${HEIGHT} --width ${WIDTH} --iter ${ITER} --epochs ${EPOCHES} --num-instances ${Inst} --eval-step ${Eval} --pooling-type ${POOL} --cluster-method ${CLUSTER}

