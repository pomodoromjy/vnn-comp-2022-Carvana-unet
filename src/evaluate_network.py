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
from generate_properties import gt_mask_process

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
    net_out_mask = torch.from_numpy(gt_mask_process(net_out_mask))
    net_out_mask = net_out_mask.to(device=device, dtype=torch.float32)
    net_out_mask.unsqueeze_(0)
    # ONNX RUNTIME
    '''
    onnx: input and output
    input: 
    img: the input image, refers to the X in the vnnlib;
    net_out_mask: the predicted mask by the model(not the user defined ground truth mask), refers to the GT in the vnnlib;
    output:
    ort_outs[0]: the logits of network output, refers to the Y_0 in the vnnlib;
    ort_outs[1]: the number of correctly predicted pixes by the onnx model, refers to the Y_1 in the vnnlib;
    '''
    inname = [input.name for input in ort_session.get_inputs()]
    outname = [output.name for output in ort_session.get_outputs()]
    ort_inputs = {inname[0]:to_numpy(img),inname[1]:to_numpy(net_out_mask)}
    ort_outs = ort_session.run(outname, ort_inputs)  # list.
    # post process.
    img_out = torch.from_numpy(ort_outs[0]).to(device=device, dtype=torch.float32)
    probs = F.softmax(img_out,dim=1)[0]
    pre_mask = F.one_hot(probs.argmax(dim=0), 2).permute(2, 0, 1)
    mask_out = pre_mask[1]
    #out_num: total numbers of correct predicted pixes
    out_num = ort_outs[1]
    print(out_num)






if __name__ == '__main__':
  test_onnx()