# @Time     :2022/4/13  8:38
# @Author   :majinyan
# @Software : PyCharm
# @Time     :2022/3/29  14:52
# @Author   :majinyan
# @Software : PyCharm
import os

import onnx

#check onnx
onnxpath = '../net/onnx/unet_simp_small.onnx'
onnx_model = onnx.load(onnxpath)
check = onnx.checker.check_model(onnx_model)
print('check: ', check)

'''
evaluate onnx
'''
import io
import torch
import torch.onnx
import onnxruntime
import numpy as np
from PIL import Image
import torch.nn.functional as F


onnxpath = '../net/onnx/unet_simp_small.onnx'
imgpath = '../dataset/test_images/'
mask_path = '../dataset/test_masks/' #grund truth masks
simp_net_pre_mask = '../dataset/succeeds_mask/unet_simp_small_pre/'
upsample_net_pre_mask = '../dataset/succeeds_mask/unet_upsample_small_pre/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def img_process(img):
    img_ndarray = np.asarray(img)
    img_ndarray = img_ndarray.transpose((2, 0, 1))
    img_ndarray = img_ndarray / 255
    return img_ndarray

def test_onnx():
  for imagename in os.listdir(simp_net_pre_mask):
    full_img = Image.open(imgpath + imagename.split('.')[0] + '.jpg')
    ort_session = onnxruntime.InferenceSession(onnxpath)
    img = torch.from_numpy(img_process(full_img))
    img = img.to(device=device, dtype=torch.float32)
    img.unsqueeze_(0)
    net_out_mask = Image.open(simp_net_pre_mask + imagename)
    net_out_mask = np.asarray(net_out_mask)
    net_out_mask = torch.from_numpy(net_out_mask)
    net_out_mask = net_out_mask.to(device=device, dtype=torch.float32)
    net_out_mask.unsqueeze_(0).unsqueeze_(0)
    # ONNX RUNTIME
    '''
    onnx: input and output
    input:
    model_input: The size of input(refers to the X in the vnnlib) is [1, 4, 31, 47], where 1 is the batchsize, 
    4 is the number of channels, 31 and 47 are the height and the width of samples respectively. 
    The first three channels represent RGB values of images. The last channel represents the model-produced mask, 
    which is used for calculating the number of correctly predicted pixes by the model.
    output:
    out_num: the number of correctly predicted pixes by the onnx model, refers to the Y in the vnnlib;
    '''
    inname = [input.name for input in ort_session.get_inputs()]
    outname = [output.name for output in ort_session.get_outputs()]
    model_input = torch.cat([img,net_out_mask],1)
    ort_inputs = {inname[0]:to_numpy(model_input)}
    # out_num: total numbers of correct predicted pixes
    out_num = ort_session.run(outname, ort_inputs)
    print(out_num)






if __name__ == '__main__':
  test_onnx()