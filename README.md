# Exploring Imagenet-a

**WARNING: this repo is very much  a work in progress**

Found cropping issue with [Imagenet-a](https://arxiv.org/pdf/1907.07174.pdf) where the preprocessing crops out the object, making classification impossible. 

This is a copy of the [original repo](https://github.com/hendrycks/natural-adv-examples) with a couple of additions:
* a "crop check" which gives the accuracy of each model when evaluated on a 10 crop of the image 
* eval and crop check files for CLIP
* UI for labeling images

### Crop Check
This simply does a 10 crop of the image and the classficiation is considered correct if any of the crop is classified correctly. 

For example:

`python clip_eval_crop.py --model RN50`


### UI

This is a UI I made to double check whether the images are valid. On the left is the original image and on the right is the image after preprocessing. 
![Crop](crop_error.png)

To run the UI, simply run `python gui_test.py`, it should allow you to specify a results file to write to, an index to skip to if you want to look at specific images, and allow you to label images. 

## Results
| Model | ImageNet Top 1 | ImageNet-a Top 1 |  ImageNet-a 10-crop Top 1 |
| ------- | ------- | ------- | ------- |
| ResNet50 | 76.13% | 0.0% | 5.44% |
| ResNet152 |  78.312% | 6.0267% | 22.0%|
| DenseNet121 |  74.434% | 2.1467% | 11.76%|
| CLIP RN50 | 59.82% | 22.76% | 45.52% |
| CLIP ViT-B/32 | 63.37% | 31.43% | 56.35% |
