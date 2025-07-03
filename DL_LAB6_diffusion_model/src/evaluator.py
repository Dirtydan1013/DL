import torch
import torch.nn as nn
import torchvision.models as models

'''===============================================================
1. Title:     

DLP Spring 2025 Lab6 classifier

2. Purpose:

For computing the classification accruacy.

3. Details:

The model is based on ResNet18 with only chaning the
last linear layer. The model is trained on iclevr dataset
with 1 to 5 objects and the resolution is the upsampled 
64x64 images from 32x32 images.

It will capture the top k highest accuracy indexes on generated
images and compare them with ground truth labels.

4. How to use

You may need to modify the checkpoint's path at line 40.
You should call eval(images, labels) and to get total accuracy.
images shape: (batch_size, 3, 64, 64)
labels shape: (batch_size, 24) where labels are one-hot vectors
e.g. [[1,1,0,...,0],[0,1,1,0,...],...]
Images should be normalized with:
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

==============================================================='''


class evaluation_model():
    def __init__(self):
        checkpoint = torch.load('./checkpoint.pth')
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.fc = nn.Sequential(
            nn.Linear(512,24),
            nn.Sigmoid()
        )
        self.resnet18.load_state_dict(checkpoint['model'])
        self.resnet18 = self.resnet18.cuda().eval()

    def compute_acc(self, out, onehot_labels):
        acc, total = 0, 0
        for i in range(out.size(0)):
            k = int(onehot_labels[i].sum().item())
            total += k
            _, outi = out[i].topk(k)
            _, li   = onehot_labels[i].topk(k)
            acc += sum(1 for j in outi if j in li)
        return acc/total

    def eval(self, images, labels):
        with torch.no_grad():
            out = self.resnet18(images)
            return self.compute_acc(out.cpu(), labels.cpu())
