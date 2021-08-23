import torch
import clip
from tqdm import tqdm
import pandas as pd

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as trn

from transformers import ViTFeatureExtractor, ViTForImageClassification

from notebooks.constants import imagenet_classes, imagenet_templates, indices_in_1k

import argparse

parser = argparse.ArgumentParser(description='Eval')
parser.add_argument('--save-results', type=str, help="file to store results")
parser.add_argument('--add', action='store_true', help="add to current results file")
parser.add_argument('--model', type=str, default="RN50", help="model to evalutate")
parser.add_argument('--dataset', type=str, default="imagenet-a", help="model to evalutate")
parser.add_argument('--preprocessing', type=str, default="CLIP", help="which preprocessing to use (CLIP or imagenet-a)")

args = parser.parse_args()

if args.model == "transformer":
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
else:
    model, preprocess = clip.load(args.model)

if args.preprocessing == "imagenet-a":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    preprocess = trn.Compose(
        [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])


if args.dataset == "imagenet":
    images = torchvision.datasets.ImageFolder("/home/lisa/data/imagenet-1000/val", transform=preprocess)
    loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)
else:
    images = dset.ImageFolder(root="~/data/imagenet-a/", transform=preprocess)
    loader = torch.utils.data.DataLoader(images, batch_size=128, shuffle=False,
                                             num_workers=4, pin_memory=True)

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


if args.model != "transformer":
    zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
    print("=> loaded zeroshot weights")

def accuracy(output, target, topk=(1,), answers=None):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    answers += [p.item() for p in pred[0]]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def get_logits(model, inp, type="CLIP"):
    if type == "transformer":
        inputs = feature_extractor(images=[i for i in inp], return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
    else:
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights

    if args.dataset == "imagenet-a":
        logits = logits[:, indices_in_1k]
    return logits


with torch.no_grad():
    answers = []
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(tqdm(loader)):
        if args.model != "transformer":
            images = images.cuda()
            target = target.cuda()

        # predict
        logits = get_logits(model, images, args.model)
        # image_features = model.encode_image(images)
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # logits = 100. * image_features @ zeroshot_weights
        # if args.dataset == "imagenet-a":
        #     logits = logits[:, indices_in_1k]

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5), answers=answers)
        top1 += acc1
        top5 += acc5
        n += images.size(0)

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100

print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")

if args.save_results:
    res = pd.DataFrame()
    if args.add:
        res = pd.read_csv(args.save_results)
    print(len(res), len(answers))
    res[f"CLIP {args.model} answers"] = answers
    res.to_csv(args.save_results, index=False)