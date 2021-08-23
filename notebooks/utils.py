import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import ast
import random
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.transforms.functional as trnF

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image

home_path = '/home/lisa'
if not os.path.isdir(home_path):
    home_path = '/Users/lisadunlap'

file = open(f'{home_path}/data/imagenet_classes.txt', 'r')
labels = ast.literal_eval(file.read())

file = open(f'{home_path}/data/imagenet-a/README.txt', 'r')
lines = file.readlines()
class_mappings = {}
for l in lines[12:-1]:
#     print(l.replace(' \n', '')[:9], l.replace(' \n', '')[10:])
    cls_nbr, cls_name = l.replace(' \n', '')[:9], l.replace(' \n', '')[10:]
    class_mappings[cls_nbr] = cls_name

image_paths = {c: [] for c in class_mappings.values()}
for root, directory, files in os.walk(f'{home_path}/data/imagenet-a'):
    for file in files:
        if '.jpg' in file:
            image_paths[class_mappings[root.split('/')[-1]]].append(root+'/'+file)

file = open(f'{home_path}/data/synset.txt', 'r')
lines = file.readlines()
imagenet_class_mappings = {}
for i, l in enumerate(lines):
    cls_nbr, cls_name = l.replace(' \n', '')[:10].replace('\'', ''), l.replace(' \n', '')[10:]
    imagenet_class_mappings[str(cls_nbr)] = i

def get_class_mappings():
    return class_mappings

def get_imagenet_labels():
    return labels

def get_image_paths():
    return image_paths

def get_imagenet_cls_mappings():
    return imagenet_class_mappings

def show_random_image(class_name = None, choice=None, verbose=True):
    if not class_name:
        class_name = random.choice(list(class_mappings.values()))
    if not choice:
        choice = random.choice(range(len(image_paths[class_name])))
    try:
        img_cls = [i for i in range(len(labels)) if class_name.lower() in labels[i].lower()][0]
    except:
        print(class_name)
    if verbose:
        print(f"Class: {class_name} \t Choice = {choice} \t Imagenet class = {img_cls}")
    return img_cls, class_name, choice, Image.open(image_paths[class_name][choice])

def get_naes_dataset():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

    naes = dset.ImageFolder(root="~/data/imagenet-a/", transform=test_transform)
    nae_loader = torch.utils.data.DataLoader(naes, batch_size=4, shuffle=False,
                                             num_workers=4, pin_memory=True)
    return naes, nae_loader

def get_displ_img(img):
    try:
        img = img.cpu().numpy().transpose((1, 2, 0))
    except:
        img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    displ_img = std * img + mean
    displ_img = np.clip(displ_img, 0, 1)
    displ_img /= np.max(displ_img)
    displ_img = displ_img
    displ_img = np.uint8(displ_img*255)
    return displ_img/np.max(displ_img)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
preprocessing = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean, std)])


def generate_cams(img_cls, image, verbose = True):
    model = models.resnet50(pretrained=True)
    target_layer = model.layer4[-1]
    input_tensor = torch.unsqueeze(preprocessing(image), 0)# Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    model.eval()
    out = model(torch.unsqueeze(preprocessing(image),0))
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    # top 5
    _, indices = torch.sort(out, descending=True)
    top5_pred = [labels[idx.item()].split(', ')[0] for idx in indices[0][:5]]
    # print(f"Top 5 = {[(labels[idx.item()].split(', ')[0], percentage[idx].item()) for idx in indices[0][:5]]}")
    if verbose:
        print("----- prediction -----")
        print(f"Class: {labels[index[0].item()]} \t Percentage: {percentage[index[0]].item()}")
        print(f"Top 5 {top5_pred}")

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    target_category = index.item()

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    pred_cam = cam(input_tensor=input_tensor, target_category=target_category)
    gt_cam = cam(input_tensor=input_tensor, target_category=img_cls)

    # In this example grayscale_cam has only one image in the batch:
    pred_cam = pred_cam[0, :]
    gt_cam = gt_cam[0, :]
    # img = np.array(crop(image))
    img = get_displ_img(torch.squeeze(input_tensor, 0))
    pred_vis = show_cam_on_image(img/np.max(img), pred_cam, use_rgb=True)
    gt_vis = show_cam_on_image(img/np.max(img), gt_cam, use_rgb=True)
    return input_tensor, top5_pred, target_category, img, pred_vis, gt_vis


def get_cam(image, model, target_cls, aug_smooth=False, eigen_smooth=False, normalize=True):
    """
    Get gradcam of IMAGE for MODEL wrt TARGET_CLS
    :param image:
    :param model:
    :param target_cls:
    :return:
    """
    target_layer = model.layer4[-1]
    input_tensor = torch.unsqueeze(preprocessing(image), 0).cuda()  # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    model.eval()
    out = model(torch.unsqueeze(preprocessing(image), 0).cuda())
    _, index = torch.max(out, 1)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    return cam(input_tensor=input_tensor, target_category=target_cls, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth, normalize=normalize)

def get_cam_tensor(image, model, target_cls, aug_smooth=False, eigen_smooth=False, normalize=True):
    """
    Get gradcam of IMAGE for MODEL wrt TARGET_CLS
    :param image:
    :param model:
    :param target_cls:
    :return:
    """
    target_layer = model.layer4[-1]
    input_tensor = torch.unsqueeze(image, 0)  # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!
    model.eval()
    out = model(torch.unsqueeze(image, 0))
    _, index = torch.max(out, 1)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layer=target_layer)

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    return cam(input_tensor=input_tensor, target_category=target_cls, aug_smooth=aug_smooth, eigen_smooth=eigen_smooth, normalize=normalize)