# LoRA
PyTorch implementation demonstrating Low Rank Adaption (LoRA) in Neural Networks, for efficient model compression and fine-tuning on domain specific tasks

## Purpose

The primary objective of this study is to train a base neural network model and subsequently fine-tune it using the Low Rank Adaptation (LoRA) technique. The fine-tuning process aims to enhance the model's accuracy. Furthermore, post-fine-tuning observations demonstrate that the added parameters do not disturb the original parameters. Instead, the analysis reveals the addition of minimal new parameters.

## Dataset

The dataset employed in this investigation is the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), a widely recognized benchmark for image classification tasks in the field of machine learning. MNIST consists of a collection of hand-written digits, encompassing images of digits from 0 to 9, each represented in a grayscale format. The choice of MNIST as the experimental dataset provides a standardized foundation for evaluating the effectiveness of low-rank adaptation techniques in the context of image recognition.

## MNIST Model

In this demonstration, a neural network architecture was crafted, encompassing a total of ~2 million parameters meticulously distributed across three linear layers. This architectural configuration was deliberately designed to facilitate a comprehensive exploration of the efficacy of low-rank adaptation techniques within the context of a neural network model with considerable complexity and parameterization.

### Model Summary

![image](https://github.com/bala1802/LoRA/assets/22103095/be629bc2-f99c-4fc0-a8b1-81c72635ade5)

## 