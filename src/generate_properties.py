# @Time     :2022/3/31  15:15
# @Author   :majinyan
# @Software : PyCharm
import argparse
import numpy as np
import os
import re
import PIL.Image as Image
import sys
import torch
import numpy.random as random

def get_random_images(path,random,length,seed):
    list = []
    for filename in os.listdir(path):
        list.append(filename.split('.')[0] + '.jpg')
    random_sel_list = []
    if random:
        np.random.seed(seed)
        random_sel_list = np.random.choice(list,length,replace=False)
    else:
        for index in range(len(list)):
            while len(random_sel_list) < length:
                random_sel_list.append(list[index])
    return random_sel_list

def write_vnn_spec(img_pre, gt_mask_pre, list, epslion, dir_path, prefix="spec", data_lb=0, data_ub=1, n_class=1,
                   mean=0.0, std=1.0, csv='',network_path='',vnnlib_path=''):
    num_input = 4 * 31 * 47
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for network in os.listdir(network_path):
        network = network.split('.')[0]
        selected_list = list[network]
        for index in range(len(selected_list)):
            imagename = selected_list[index]
            for eps in epslion:
                spec_name = f"{prefix}_{network}_idx_{imagename.split('.')[0]}_eps_{eps:.5f}.vnnlib"
                spec_path = os.path.join(dir_path, spec_name)
                x = Image.open(img_pre + imagename)
                x = np.array(x) / 255
                x_lb = np.clip(x - eps, data_lb, data_ub)
                x_lb = ((x_lb-mean)/std).reshape(-1)
                x_ub = np.clip(x + eps, data_lb, data_ub)
                x_ub = ((x_ub - mean) / std).reshape(-1)

                gt_mask = Image.open(gt_mask_pre[network] + '/' + imagename.split('.')[0]+'.gif')
                gt_mask = np.asarray(gt_mask)
                gt_mask = gt_mask.reshape(-1)

                with open(spec_path, "w") as f:
                    f.write(f"; Spec for sample id {imagename} and epsilon {eps:.5f}\n")

                    f.write(f"\n; Definition of input variables(image and model predicted mask)\n")
                    for i in range(num_input):
                        f.write(f"(declare-const X_{i} Real)\n")

                    f.write(f"\n; Definition of output variables\n")
                    for i in range(n_class-1):
                        f.write(f"(declare-const Y_{i} Real)\n")

                    f.write(f"\n; Definition of input constraints(image and model predicted mask)\n")
                    for i in range(num_input):
                        if i < len(x_ub):
                            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
                            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")
                        else:
                            f.write(f"(assert (= X_{i} {gt_mask[i-len(x_ub)]:.8f}))\n")

                    f.write(f"\n; Definition of output constraints\n")
                    for i in range(n_class-1):
                        f.write(f"(assert (<= Y_{i} 1312))\n")
    #make csv file
    if not os.path.exists(csv):
        os.system(r"touch {}".format(csv))
    csvFile = open(csv, "w")
    timeout = 300
    for vnnLibFile in os.listdir(vnnlib_path):
        net1 = "unet_simp_small"
        net2 = "unet_upsample_small"
        if "unet_simp_small" in vnnLibFile:
            print(f"{net1},{vnnLibFile},{timeout}", file=csvFile)
        else:
            print(f"{net2},{vnnLibFile},{timeout}", file=csvFile)
    csvFile.close()


def main():
    seed = int(sys.argv[1])
    mean = 0.0
    std = 1.0
    epsilon = [0.012,0.015]
    csv = "../Carvana-unet_instances.csv"

    '''get the list of success images'''
    sucess_images_mask = {'unet_simp_small':'../dataset/succeeds_mask/unet_simp_small_pre',
                          'unet_upsample_small':'../dataset/succeeds_mask/unet_upsample_small_pre'}
    unet_simp_small_list = get_random_images(sucess_images_mask['unet_simp_small'],random=True,length=40,seed=seed)
    unet_upsample_small_list = get_random_images(sucess_images_mask['unet_upsample_small'],random=True,length=40,seed=seed)
    list = {'unet_simp_small':unet_simp_small_list,'unet_upsample_small':unet_upsample_small_list}
    network_path = '../net/onnx/'
    vnnlib_path = '../specs/vnnlib/'
    img_file_pre = r'../dataset/test_images/'
    mean = np.array(mean).reshape((1, -1, 1, 1)).astype(np.float32)
    std = np.array(std).reshape((1, -1, 1, 1)).astype(np.float32)
    write_vnn_spec(img_file_pre, sucess_images_mask, list, epsilon, dir_path='../specs/vnnlib', prefix='spec',
                   data_lb=0,
                        data_ub=1, n_class=2, mean=mean, std=std,csv=csv,
                       network_path=network_path, vnnlib_path=vnnlib_path)


if __name__ == "__main__":
    main()