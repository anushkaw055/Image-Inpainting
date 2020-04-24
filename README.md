# Image-Inpainting
# Partial Convolutions for Image Inpainting using Keras
Keras implementation of "*Image Inpainting for Irregular Holes Using Partial Convolutions*", https://arxiv.org/abs/1804.07723. 

# Dependencies
* Python 3.6
* Keras 2.2.4
* Tensorflow 1.12

The primary implementations of the new `PConv2D` keras layer as well as the `UNet`-like architecture using these partial convolutional layers can be found in jupyter notebook

Step 1: Creating random irregular masks<br />
Step 2: Implementing and testing the implementation of the `PConv2D` layer<br />
Step 3: Implementing and testing the UNet architecture with `PConv2D` layers<br />
Step 4: Training & testing the final architecture on ImageNet<br />
Step 5: Simplistic attempt at predicting arbitrary image sizes through image chunking

## Pre-trained weights
I've ported the VGG16 weights from PyTorch to keras; this means the `1/255.` pixel scaling can be used for the VGG16 network similarly to PyTorch. 
* [Ported VGG 16 weights](https://drive.google.com/open?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0)

# Implementation details
Details of the implementation are in the [paper itself](https://arxiv.org/abs/1804.07723), however I'll try to summarize some details here.

## Mask Creation
In the paper they use a technique based on occlusion/dis-occlusion between two consecutive frames in videos for creating random irregular masks - instead I've opted for simply creating a simple mask-generator function which uses OpenCV to draw some random irregular shapes which I then use for masks. Plugging in a new mask generation technique later should not be a problem though, and I think the end results are pretty decent using this method as well.

## Partial Convolution Layer
A key element in this implementation is the partial convolutional layer. Basically, given the convolutional filter **W** and the corresponding bias *b*, the following partial convolution is applied instead of a normal convolution:

<img src='./data/images/eq1.PNG' />

where ⊙ is element-wise multiplication and **M** is a binary mask of 0s and 1s. Importantly, after each partial convolution, the mask is also updated, so that if the convolution was able to condition its output on at least one valid input, then the mask is removed at that location, i.e.

<img src='./data/images/eq2.PNG' />

The result of this is that with a sufficiently deep network, the mask will eventually be all ones (i.e. disappear)

## UNet Architecture
Specific details of the architecture can be found in the paper, but essentially it's based on a UNet-like structure, where all normal convolutional layers are replace with partial convolutional layers, such that in all cases the image is passed through the network alongside the mask. The following provides an overview of the architecture.
<img src='./data/images/architecture.png' />

## Loss Function(s)
The loss function used in the paper is kinda intense, and can be reviewed in the paper. In short it includes:

* Per-pixel losses both for maskes and un-masked regions
* Perceptual loss based on ImageNet pre-trained VGG-16 (*pool1, pool2 and pool3 layers*)
* Style loss on VGG-16 features both for predicted image and for computed image (non-hole pixel set to ground truth)
* Total variation loss for a 1-pixel dilation of the hole region
