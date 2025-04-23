### A Principle Design of Registration-Fusion Consistency: Towards Interpretable Deep Fusion for Unregistered Hyperspectral Image



The python code implementation of the paper "A Principle Design of Registration-Fusion Consistency: Towards Interpretable Deep Fusion for Unregistered Hyperspectral Image" (TNNLS 2024)

### Requirements
    Ubuntu 18.04 cuda 11.0
    Python 3.7 Pytorch 1.11
    
To install requirements:
        pip install -r requirements.txt
### Usage
## Brief description
The train.py include training code and some parameters, detailed in train.py.

The model.py include main model structure.

## dataset
    the default dataset path can be change in dataloader.py
          ─pavia
            ├─train
            │  ├─gtHS
            │  │  ├─1.mat
            │  │  ├─2.mat
            │  │  └─3.mat
            │  ├─hrMS
            │  ├─LRHS
            │  └─LRHS_**
            └─test

## training
    To do training with MoE-PNP for 80 epochs on pavia with elastic distortions, run

```
python train.py --data /dataset/pavia \
  --dataset pavia --type _Elastic600\
  --lr 0.0001 -p 200 --epochs 80
```


## testing
    To evaluate the performance on the test set, run
    
```
python test.py 
```



