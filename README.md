# da
PyTorch implementation for **Prototype-Based Multisource Domain Adaptation** (TNNLS2021). This repository is based on framework from [dassl](https://github.com/KaiyangZhou/Dassl.pytorch) and modified part of the code. 

The installation can refer to dassl.

Please download dig-5, office-31, office-home to datasets folder.

## Training:
```
CUDA_VISIBLE_DEVICES=1 python tools/train.py --trainer MSDTR_CDAN --source-domains mnist mnist_m svhn syn --target-domains usps  --dataset-config-file configs/datasets/digit5.yaml --config-file configs/trainers/msdtr_cdan/digit5.yaml --output-dir output_final/msdtr_cdan/dig/usps/1
```



## Citation
If you use this code for your research, please cite our [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9404309):

```
@article{zhou2021prototype,
  title={Prototype-Based Multisource Domain Adaptation},
  author={Zhou, Lihua and Ye, Mao and Zhang, Dan and Zhu, Ce and Ji, Luping},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2021},
  publisher={IEEE}
}
```
