## Model

생성모델을 사용하여 non-robust feature에서 원래 feature로 생성할 수 있도록 하였다.

생성 모델을 활용하면, non-robust feature을 original feature와 비슷하게 생성하여 적대적 예제에도 강건한 모델을 생성할 수 있다.

![MODEL_FIGURE](/figures/figure.png "framework")

## Setup

```
conda env create -n FSRGAN python=3.7
conda activate FSRGAN
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install tqdm
```

## Train

```
python train.py --save_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 0 --epoch 200 --bs 256
```

## Test

```
python test.py --load_name cifar10_vgg16 --dataset cifar10 --model vgg16 --device 0
```

## Reference

https://github.com/wkim97/FSR

https://github.com/sky4689524/DefenseGAN-Pytorch
