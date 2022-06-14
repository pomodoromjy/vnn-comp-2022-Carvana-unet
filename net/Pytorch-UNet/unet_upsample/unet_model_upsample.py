""" Full assembly of the parts to form the complete network """
import torch

from unet_upsample.unet_parts_upsample import *
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
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down4 = DoubleConv(128, 128)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def my_softmax(self,x):
        return F.softmax(x,dim=1)[0]

    def my_ont_hot(self,x):
        return F.one_hot(x.argmax(dim=0), 2).permute(2, 0, 1)

    def get_num(self,pre_mask,original_gt):
        in1 = pre_mask[1].int()
        in2 = original_gt.squeeze().int()
        return torch.eq(in1, in2).sum()

    def forward(self, x,original_gt):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x5 = self.down4(x2)
        x6 = self.up4(x5,x1)
        logits = self.outc(x6)
        print(logits.shape)
        # logits one-hot encoding
        probs = self.my_softmax(logits)
        pre_mask = self.my_ont_hot(probs)
        #number of pixes correctly predicted
        num = self.get_num(pre_mask.to(device=device),original_gt.to(device=device))
        return (logits,num)



if __name__ == "__main__":
    input_img = torch.randn(1, 3, 31, 47)
    input_gt_mask = torch.randn(1, 31, 47)
    model = UNet_simp(n_channels=3, n_classes=2, bilinear=False)
    output = model(input_img,input_gt_mask)
    print("out:",output)

