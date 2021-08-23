import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import pandas as pd

from calibration_tools import *
from notebooks.constants import indices_in_1k

import argparse

parser = argparse.ArgumentParser(description='Eval')
parser.add_argument('--save-results', type=str, help="file to store results") ## TODO: finish
parser.add_argument('--add', action='store_true', help="add to current results file")
parser.add_argument('--model', type=str, default="resnet50", help="model to evalutate")

args = parser.parse_args()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# original preprocessing
# test_transform = trn.Compose(
#     [trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

test_transform = trn.Compose([trn.Resize(256),
                            trn.TenCrop(224), # this is a list of PIL Images
                            trn.Lambda(lambda crops: torch.stack([trn.ToTensor()(crop) for crop in crops])),
                            trn.Normalize(mean, std)
                           ])

naes = dset.ImageFolder(root="~/data/imagenet-a/", transform=test_transform)
nae_loader = torch.utils.data.DataLoader(naes, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)

net = getattr(models, args.model)(pretrained=True)
# net.load_state_dict(torch.load("pretrained_models/resnet152_cutmix_acc_80_80.pth"))

net.cuda()
net.eval()


concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to('cpu').numpy()

def get_net_results():
    confidence = []
    correct = []

    possible_errors = []

    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(nae_loader):
            data, target = torch.squeeze(data, 0).cuda(), target.cuda()
            target_dup =torch.squeeze(torch.stack([target for i in range(data.size()[0])]).cuda(), 1)

            output = net(data)[:,indices_in_1k]

            # accuracy
            pred = output.data.max(1)[1]
            crop_corr = pred.eq(target_dup.data).sum().item()
            num_correct += np.min([crop_corr, 1.0])
            if pred[4].eq(target.data).sum().item() == 0.0 and crop_corr > 0.0:
                possible_errors.append([batch_idx, crop_corr, target[0].item(), pred[4].item()])

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

    pd.DataFrame(possible_errors, columns=["error", "crop corr", "target", "pred"]).to_csv('results/possible_errors_resnet152_cutmix_fivecrop.csv', index=False)

    return num_correct / len(nae_loader.dataset), confidence.copy(), correct.copy()


acc, test_confidence, test_correct = get_net_results()

print('ImageNet-A Accuracy (%):', round(100*acc, 4))

show_calibration_results(np.array(test_confidence), np.array(test_correct))


