import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
# from scipy.ndimage import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    return dice
    # , hd95
    # , asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    
    h, w = 224, 224
    
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (h / x, w / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)

            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / h, y / w), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    
    potential_m1 = os.path.join(snapshot_path, f'{FLAGS.model}_best_model1.pth')
    if FLAGS.model == "mambaunet" and os.path.exists(potential_m1):
        model_suffixes = ["_best_model1.pth", "_best_model2.pth"]
    else:
        model_suffixes = ["_best_model.pth"]
    
    all_results = {}
    
    for suffix in model_suffixes:
        test_save_path = os.path.join(snapshot_path, "{}{}_predictions".format(FLAGS.model, suffix.replace('.pth','')))
        test_save_path += "/"
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path)

        # 1. 实例化网络
        net = net_factory(net_type=FLAGS.model, in_chns=1,
                          class_num=FLAGS.num_classes)
        
        # 2. 拼接权重路径
        save_mode_path = os.path.join(snapshot_path, '{}{}'.format(FLAGS.model, suffix))
        
        # 检查权重文件是否存在，不存在则跳过（防止普通 mambaunet 只有单权重时报错）
        if not os.path.exists(save_mode_path):
            print(f"Skipping {save_mode_path}, file not found.")
            continue

        print(f"\n>>>> Testing: {save_mode_path}")
        net.load_state_dict(torch.load(save_mode_path))
        print("init weight from {}".format(save_mode_path))
        net.eval()

        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        
        for case in tqdm(image_list):
            first_metric, second_metric, third_metric = test_single_volume(
                case, net, test_save_path, FLAGS)
            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)
            third_total += np.asarray(third_metric)
        
        avg_metric = [first_total / len(image_list), 
                      second_total / len(image_list), 
                      third_total / len(image_list)]
        
        # 打印当前权重的平均结果
        print(f"Result for {suffix}: {avg_metric}")
        all_results[suffix] = avg_metric

    # 为了兼容原来的调用处，我们返回最后一个跑完的指标，或者你可以根据需要修改
    return avg_metric

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
