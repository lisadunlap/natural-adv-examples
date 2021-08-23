import torch
import clip
from tqdm.notebook import tqdm
import pandas as pd

import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF

from notebooks.constants import imagenet_classes, imagenet_templates, indices_in_1k

import argparse

parser = argparse.ArgumentParser(description='Eval')
parser.add_argument('--save-results', type=str, help="file to store results")
parser.add_argument('--add', action='store_true', help="add to current results file")
parser.add_argument('--model', type=str, default="RN50", help="model to evalutate")

args = parser.parse_args()

model, preprocess = clip.load(args.model)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


test_transform = trn.Compose(
    [trn.Resize(size=256, interpolation=trnF.InterpolationMode.BICUBIC),
     trn.TenCrop(224),  # this is a list of PIL Images
     trn.Lambda(lambda crops: torch.stack([trn.ToTensor()(crop) for crop in crops])),
     trn.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])


naes = dset.ImageFolder(root="~/data/imagenet-a/", transform=test_transform)
nae_loader = torch.utils.data.DataLoader(naes, batch_size=1, shuffle=False,
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


zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)
print("=> loaded zeroshot weights")

def accuracy(output, target, topk=(1,), answers=None):
    pred = output.topk(max(topk), 1, True, True)[1].t()
#     if answers:
    answers += [p.item() for p in pred[0]]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    a1, a5 = 0.0, 0.0
    # for k in topk:
    #     if float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) > 0.0 and float(correct[:k][:, 4].float().sum(0, keepdim=True).cpu().numpy()) == 0.0:
    #         print("incorrect crop")

    return [min(1.0, float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())) for k in topk]


with torch.no_grad():
    answers = []
    top1, top5, n = 0., 0., 0.
    for i, (images, target) in enumerate(nae_loader):
        images = torch.squeeze(images, 0).cuda()
        target = target.cuda()
        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100. * image_features @ zeroshot_weights
        logits = logits[:, indices_in_1k]

        # measure accuracy
        acc1, acc5 = accuracy(logits, target, topk=(1, 5), answers=answers)
        top1 += acc1
        top5 += acc5
        n += 1

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