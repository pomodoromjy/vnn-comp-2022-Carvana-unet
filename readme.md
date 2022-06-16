**A  set of simplified Unet Benchmarks on Carvana for Neural Network Verification**

We propose a new set of benchmarks of simplified unet on Carvana for neural network verification in this repository.

**Motivation**

Currently, most networks evaluated in the literature are focusing on image classfication. However, in many practical scenarios, e.g. autonomous driving, people pay more attention to object detection or semantic segmentation. Considering the complexity of the object detection, we propose a new set of simplified Unet (model one: four Conv2d layers followed with BN and ReLu; model two: add one AveragePool layer and one  TransposedConv Upsampleing layer on the model one ). We advocate that tools should handle more practical architectures, and the simplified Unet is the first step towards this goal.

**Model details**

The ONNX format networks are available in the *[onnx] (net/onnx/)* folder, and the pytorch models can be found in the *[pytorch] (net/pytorch/)*  folder. And the inference script(`evaluate_network.py`) of onnx models can be found in the *[src] (src/)* folder.

**Data Format**

The input images should be normalized to the [0, 1] range. The ground truth masks, which is produced by executing the model on original images,  is either 0 or 1 for each pixel.

**Data Selection**

The Carvana dataset consists of 5088 images covering 318 cars, which means each car has 16 images. We choose one image for each car, 318 images in total, as a testset. And the remaining 4700 images are used for training. There are 52 images  whose 98.8 percent pixes can be predicted correctly for model one, and 44 images whose 99 percent pixes can be predicted correctly for model two.  We propose to randomly select 16 images from these images for verification.

**More details**

- The input size is [1, 4, 31, 47], where 1 is the batchsize, 4 is the number of channels, 31and 47 are the height and the width of samples respectively. The first three channels represent RGB values of images. The last channel represents the model-produced mask, which is used for calculating the number of correctly predicted pixes by the model.
- The model has one output, which is the number of correctly predicted pixes by the model, compared with the model-produced mask.
- The PyTorch model has two outputs, which are the logits and the matched number. We delete logits in the onnx model so that we have the file which fits the vnnlib format.
- The image and mask size are both 47x31 to alleviate computational burden.
- The .vnnlib and .csv files were created with `generate_properties.py` script, which can be found in  *[src] (src/)*  folder.
- The Carvana-unet_instances.csv containts the full list of benchmark instances, one per line: onnx_file, vnn_lib_file, timeout_secs.