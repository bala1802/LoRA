# LoRA
PyTorch implementation demonstrating Low Rank Adaption (LoRA) in Neural Networks, for efficient model compression and fine-tuning on domain specific tasks

## Dataset

The dataset employed in this investigation is the MNIST dataset (http://yann.lecun.com/exdb/mnist/), a widely recognized benchmark for image classification tasks in the field of machine learning. MNIST consists of a collection of hand-written digits, encompassing images of digits from 0 to 9, each represented in a grayscale format. The choice of MNIST as the experimental dataset provides a standardized foundation for evaluating the effectiveness of low-rank adaptation techniques in the context of image recognition.

## MNIST Model

In this demonstration, a neural network architecture was crafted, encompassing a total of 2 million parameters meticulously distributed across three linear layers. This architectural configuration was deliberately designed to facilitate a comprehensive exploration of the efficacy of low-rank adaptation techniques within the context of a neural network model with considerable complexity and parameterization.

### Model Summary

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                 [-1, 1000]         785,000
              ReLU-2                 [-1, 1000]               0
            Linear-3                 [-1, 2000]       2,002,000
              ReLU-4                 [-1, 2000]               0
            Linear-5                   [-1, 10]          20,010
              ReLU-6                   [-1, 10]               0
================================================================
Total params: 2,807,010
Trainable params: 2,807,010
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 10.71
Estimated Total Size (MB): 10.76
----------------------------------------------------------------
