""" Full assembly of the parts to form the complete network """
import torch

from unet_simp.unet_parts_simp import *
from torchvision import transforms
from PIL import Image
from utils.data_loading import BasicDataset
import numpy as np



tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((31, 47)),
                transforms.ToTensor()
            ])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UNet_simp(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_simp, self).__init__()
        self.n_channels = n_channels-1
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down4 = DoubleConv(128, 64)
        self.outc = OutConv(64, n_classes)

    def my_softmax(self,x):
        return F.softmax(x,dim=1)[0]

    def my_ont_hot(self,x):
        return F.one_hot(x.argmax(dim=0), 2).permute(2, 0, 1)

    def get_num(self,pre_mask,original_gt):
        in1 = pre_mask[1].int()
        in2 = original_gt.int()
        return torch.eq(in1, in2).sum()

    def forward(self, input):
        x = input[:, :3]
        gt = input[:, -1]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x5 = self.down4(x2)
        logits = self.outc(x5)
        print(logits.shape)
        # logits one-hot encoding
        probs = self.my_softmax(logits)
        pre_mask = self.my_ont_hot(probs)
        # number of pixes correctly predicted
        num = self.get_num(pre_mask.to(device=device), gt.to(device=device))
        return (logits, num)



if __name__ == "__main__":
    input_img = torch.randn(1, 3, 31, 47)
    input_gt_mask = torch.randn(1, 1, 31, 47)
    input = torch.cat([input_img, input_gt_mask], 1)
    model = UNet_simp(n_channels=4, n_classes=2, bilinear=False)
    output = model(input)
    print("out:", output)