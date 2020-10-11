# Introduction
This is the source code of our TMM 2020 paper "Deep Reinforcement Learning for Image Hashing". Please cite the following paper if you use our code.

Yuxin Peng, Jian Zhang and Zhaoda Ye, "Deep Reinforcement Learning for Image Hashing", IEEE Transactions on Multimedia (TMM), Vol. 22, No. 8, pp. 2061-2073, Aug. 2020.(SCI, EI)

# Dependency

This code is implemented with pytorch.

# Data Preparation

The codes takes the images of cifar10 as the input.

The images are organized as follows:

./train/class_id/name_of_image

./test/class_id/name_of_image

# Usage

Start training and tesing by executiving the following commands. This will train and test the model on Cifar10 datatset. 

python train.py 0 32 16 cifar

train.py [gpu_id,bit_length,batch_size,dataset]


