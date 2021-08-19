# densityclustering
Unsupervised Learning for Fine-grained Image Classification

![Python >=3.5](https://img.shields.io/badge/Python->=3.6-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-yellow.svg)

# Bridging the Gap Between Supervised and Unsupervised Learning for Fine-grained Image Classification

The *official* repository for [Bridging the Gap Between Supervised and Unsupervised Learning for Fine-grained Image Classification](https://arxiv.org/abs/2108.xxxxxx).

## Requirements

### Installation

```shell
git clone https://github.com/jiabaowang/densityclustering.git
cd densityclustering
python setup.py develop
```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the FGVC datasets CUB-200-2011, Stanford-Dogs, Stanford-Cars, FGVC-Aircraft, Oxford-Flowers and Oxford-Pets.
Then unzip them under the directory like

```
DensityClustering/examples/data
├── aircraft
│   └── FGVC_Aircraft
├── cars
│   └── Stanford-Cars
├── cub200
│   └── CUB-200-2011
├── dogs
│   └── Stanford-Dogs
├── flowers
│   └── Oxford_Flowers
└── pets
    └── Oxford_Pets
```

## Training

We utilize 4 GTX-1080TI GPUs for training. For more parameter configuration, please check **`train_fgvc_cub.sh`**.

**examples:**

CUB-200-2011:

```shell
./train_fgvc_cub.sh
```

## Evaluation

We utilize 4 GTX-1080TI GPU for testing.

**examples:**

CUB-200-2011:
```shell
### CUB-200-2011 ###
./test_fgvc_cub.sh
```

## Results

You can download the trained models in the paper from [BaiduYun](https://) 

CUB-200-2011

| Method | Top-1(%)	| ACC(%)	| NMI(%)	| ARI(%) |
|---------|---------|---------|---------|---------|
| DensityClustering | 71.5 | 59.1 | 78.5 | 37.7 |


# Acknowledgements

Thanks for the excellent opening sources
[SpCL](https://github.com/yxgeee/SpCL), 
[GroupSampling](https://github.com/ucas-vg/GroupSampling), 
[ClusterContrast](https://github.com/alibaba/cluster-contrast-reid), 
[ICE](https://github.com/chenhao2345/ICE).

