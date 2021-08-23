# Exploring Imagenet-a

**WARNING: this repo is very much  a work in progress**

Found cropping issue with [Imagenet-a](https://arxiv.org/pdf/1907.07174.pdf) where the preprocessing crops out the object, making classification impossible. 

This is a copy of the [original repo](https://github.com/hendrycks/natural-adv-examples) with a couple of additions:
* a "crop check" which gives the accuracy of each model when evaluated on a 10 crop of the image 
* eval and crop check files for CLIP
* UI for labeling images

### Crop Check
This simply does a 10 crop of the image and the classficiation is considered correct if any of the crop is classified correctly. 

`python clip_eval_crop.py --model RN50`


### UI
This is a UI I made to double check whether the images are valid. On the left is the original image and on the right is the image after preprocessing. 
![Crop](crop_error.png)

## Results
| Model | ImageNet Top 1 | ImageNet-a Top 1 |  ImageNet-a 10-crop Top 1 |
| ------- | ------- | ------- | ------- |
| ResNet50 | 76.13% | 0.0% | 5.44% |
| ResNet152 |  78.312% | 6.0267% | 22.0%|
| DenseNet121 |  74.434% | 2.1467% | 11.76%|
| CLIP RN50 | 59.82% | 22.76% | 39.97% |
| CLIP ViT-B/32 | 63.37% | 31.43% | 51.37% |


# Original repo: Natural Adversarial Examples

We introduce [natural adversarial examples](https://arxiv.org/abs/1907.07174) -- real-world, unmodified, and naturally occurring examples that cause machine learning model performance to significantly degrade.

__[Download the natural adversarial example dataset ImageNet-A for image classifiers here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar).__

__[Download the natural adversarial example dataset ImageNet-O for out-of-distribution detectors here](https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar).__

<img align="center" src="examples.png" width="400">
Natural adversarial examples from ImageNet-A and ImageNet-O. The black text is the actual class, and
the red text is a ResNet-50 prediction and its confidence. ImageNet-A contains images that classifiers should be
able to classify, while ImageNet-O contains anomalies of unforeseen classes which should result in low-confidence
predictions. ImageNet-1K models do not train on examples from “Photosphere” nor “Verdigris” classes, so these images
are anomalous. Many natural adversarial examples lead to wrong predictions, despite having no adversarial modifications as they are examples which occur naturally.

## Citation

If you find this useful in your research, please consider citing:

    @article{hendrycks2021nae,
      title={Natural Adversarial Examples},
      author={Dan Hendrycks and Kevin Zhao and Steven Basart and Jacob Steinhardt and Dawn Song},
      journal={CVPR},
      year={2021}
    }
