
## UDTL-based-Intelligent-Diagnosis



This repository contains code that applies DA, UDA, and DG methodologies for fault diagnosis of rotating machinery.

It is planned to be updated step by step.

If there are any errors or incorrect code, please let me know.

This repository contains the implementation details of our paper: [IEEE Transactions on Instrumentation and Measurement] **[Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study](https://ieeexplore.ieee.org/document/9552620)** by [Zhibin Zhao](https://zhaozhibin.github.io/), Qiyang Zhang, and Xiaolei Yu.
The methods about multi-domain TL can be found in (https://github.com/zhanghuanwang1/UDTL_multi_domain)
## Correction
* 

## Guide


## Requirements
- Python 3.6.9
- Numpy 1.19.5
- Pandas 1.1.5
- tqdm 4.62.3
- Scipy 1.2.1
- pytorch >= 1.1
- torchvision >= 0.40


## Datasets
- **[CWRU Bearing Dataset](https://csegroups.case.edu/bearingdatacenter/pages/download-data-file/)**
- **[JNU Bearing Dataset](http://mad-net.org:8765/explore.html?t=0.5831516555847212.)**

## References

Part of the code refers to the following open source code:
- [CORAL.py](https://github.com/SSARCandy/DeepCORAL) from the paper "[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35)" proposed by Sun et al.
- [DAN.py and JAN.py](https://github.com/thuml/Xlearn) from the paper "[Deep Transfer Learning with Joint Adaptation Networks](https://dl.acm.org/citation.cfm?id=3305909)" proposed by Long et al.
- [AdversarialNet.py and entropy_CDA.py](https://github.com/thuml/CDAN) from the paper "[Conditional adversarial domain adaptation](http://papers.nips.cc/paper/7436-conditional-adversarial-domain-adaptation)" proposed by Long et al.


## Pakages

This repository is organized as:
- [loss](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/loss) contains different loss functions for Mapping-based DTL.
- [datasets](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/datasets) contains the data augmentation methods and the Pytorch datasets for time and frequency domains.
- [models](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/models) contains the models used in this project.
- [utils](https://github.com/ZhaoZhibin/DL-based-Intelligent-Diagnosis-Benchmark/tree/master/utils) contains the functions for realization of the training procedure.


## Usage
- download datasets
- use the train_base.py to test Basis and AdaBN (network-based DTL and instanced-based DTL)

- for example, use the following commands to test Basis for CWRU with the transfer_task 0-->1
- `python train_advanced.py --data_name CWRU --data_dir D:/Data --method "base" --transfer_task [0],[1] --adabn ""`
- for example, use the following commands to test AdaBN for CWRU with the transfer_task 0-->1
- `python train_advanced.py --data_name CWRU --data_dir D:/Data --method "base" --transfer_task [0],[1] --adabn "True"`

- use the train_advanced.py to test (mapping-based DTL and adversarial-based DTL)
- for example, use the following commands to test DANN for CWRU with the transfer_task 0-->1
- `python train_advanced.py --data_name CWRU --data_dir D:/Data --transfer_task [0],[1] --method "DA"  --last_batch "" --distance_metric "" --domain_adversarial True --adversarial_loss DA`
- for example, use the following commands to test MK-MMD for CWRU with the transfer_task 0-->1
- `python train_advanced.py --data_name CWRU --data_dir D:/Data --transfer_task [0],[1] --method "DA" --last_batch True --distance_metric True --distance_loss MK-MMD --domain_adversarial "" `

- Multi DA and DG will be updated.


## Citation
Codes:
```
@misc{Zhao2019,
author = {Zhibin Zhao and Qiyang Zhang and Xiaolei Yu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
title = {Unsupervised Deep Transfer Learning for Intelligent Fault Diagnosis},
year = {2019},
publisher = {GitHub},
journal = {GitHub repository},
howpublished = {\url{https://github.com/ZhaoZhibin/UDTL}},
}
```
Paper:
```
@article{zhao2021applications,
  title={Applications of Unsupervised Deep Transfer Learning to Intelligent Fault Diagnosis: A Survey and Comparative Study},
  author={Zhibin Zhao and Qiyang Zhang and Xiaolei Yu and Chuang Sun and Shibin Wang and Ruqiang Yan and Xuefeng Chen},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2021}
}
```
## Contact
- dsym2894@yonsei.ac.kr
