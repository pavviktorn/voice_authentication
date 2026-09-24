#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import model
from detection_layers.modules import MultiBoxLoss
from dataset import DeepfakeDataset
from lib.util import load_config, update_learning_rate, my_collate, get_video_auc
import cv2

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/caddm_test.cfg')
    args = parser.parse_args()
    return args


def load_checkpoint(ckpt, net, device):
    checkpoint = torch.load(ckpt)

    gpu_state_dict = OrderedDict()
    for k, v in checkpoint['network'] .items():
        name = "module." + k  # add `module.` prefix
        name = k
        gpu_state_dict[name] = v.to(device)
    net.load_state_dict(gpu_state_dict)
    return net


def test():

    torch.set_flush_denormal(True)

    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    net = net.to(device)
    # net = nn.DataParallel(net)
    net.eval()
    if cfg['model']['ckpt']:
        net = load_checkpoint(cfg['model']['ckpt'], net, device)

    #############################################################
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(net, dummy_input, "out.onnx", keep_initializers_as_inputs=False, verbose=False,
                      opset_version=12)

    import onnx
    onnx_model = onnx.load("out.onnx")
    from onnxsim import simplify
    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    import onnxoptimizer
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, "out.onnx")

    from onnx import numpy_helper
    total_parameters = 0
    for initializer in onnx_model.graph.initializer:
        total_parameters += numpy_helper.to_array(initializer).size

    # model_cv = cv2.dnn.readNet("out.onnx")
    # model_cv.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # input_mean = 0
    # input_std = 1
    # imgs = cv2.imread('./test1.jpg')
    # imgs = [imgs]
    # blob = cv2.dnn.blobFromImages(imgs, 1.0 / input_std, (224,224),
    #                               (input_mean, input_mean, input_mean), swapRB=True)
    # model_cv.setInput(blob, "input.1")
    # net_out = model_cv.forward("466")
    ############################################################



    # get testing data
    print(f"Load deepfake dataset from {cfg['dataset']['img_path']}..")
    test_dataset = DeepfakeDataset('test', cfg, cfg['dataset']['info_path'])
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=False, num_workers=0,#zzzz
                             )

    # start testing.
    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    for batch_data, batch_labels in test_loader:

        labels, video_name = batch_labels
        labels = labels.long()

        outputs = net(batch_data)
        outputs = outputs[:, 1]
        frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
        frame_label_list.extend(labels.detach().cpu().numpy().tolist())
        video_name_list.extend(list(video_name))

        print(f"{video_name[0]} label:{labels.detach().cpu().numpy()[0]} out:{outputs.detach().cpu().numpy()[0]}")

    f_auc = roc_auc_score(frame_label_list, frame_pred_list)
    v_auc = get_video_auc(frame_label_list, video_name_list, frame_pred_list)
    print(f"Frame-AUC of {cfg['dataset']['name']} is {f_auc:.4f}")
    print(f"Video-AUC of {cfg['dataset']['name']} is {v_auc:.4f}")


if __name__ == "__main__":
    test()

# vim: ts=4 sw=4 sts=4 expandtab
