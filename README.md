# ANN-to-SNN-for-Inception-ResNet
This is a project for the proposed Homeostasis-based ANN-to-SNN conversion for Inception and Residual architecture. This work has been 
summarized in paper *Homeostasis-based ANN-to-SNN Conversion of Inception and Residual Architecture for Object Classification* submitted to
IEEE ICASSP-2019. 
The CNN architectures that has been converted in this project includes:
* VGG-16
* ResNet-20,32,44,56
* Inception-v4
* Inception-ResNet-v2

Take the conversion of Inception as an example, three parts are included:
* pre-train CNN models:
  * network definition: *inception_init.m*
  * training CNNs: *train_cifar.m*
* parse Inception(dagnn format): *parse_dagnn.m*
  * normalize weights
  * construct SNN
* simulate SNN: *dagnn2snn.m*
* run script: *conversion_dagnn.m*

The training of CNNs and the simulation of SNN all are implemented on Matlab with Deep Learning Library: Matconvnet.
