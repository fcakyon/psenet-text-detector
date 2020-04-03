import os
import cv2
import collections
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

from psenet_text_detector.models import resnet50
from psenet_text_detector.pypse import pse as pypse
from psenet_text_detector.utils import download, scale_image


def copy_state_dict(state_dict):
    new_state_dict = collections.OrderedDict()
    for key, value in state_dict.items():
        tmp = key[7:]
        new_state_dict[tmp] = value
    return new_state_dict


def load_psenet_model(cuda: bool = False,
                      scale: int = 1):
    # get psenet net path
    home_path = str(Path.home())
    weight_path = os.path.join(home_path,
                               "psenet_text_detector",
                               "weights",
                               "psenet_best.pth")
    # load base resnet model
    model = resnet50(pretrained=False, num_classes=7, scale=scale)  # initialize

    # check if weights are already downloaded, if not download
    url = "https://drive.google.com/uc?id=1w2IgRkQXX49AbOARitO5xCr8-N93JHDd"
    if os.path.isfile(weight_path) is not True:
        print("PSENet text detector weight will be downloaded to {}"
              .format(weight_path))

        download(url=url, save_path=weight_path)

    # arange device
    if cuda:
        checkpoint = torch.load(weight_path, map_location='cuda')
        model.load_state_dict(copy_state_dict(checkpoint['state_dict']))

        model = model.cuda()
        cudnn.benchmark = False
    else:
        checkpoint = torch.load(weight_path, map_location='cpu')
        model.load_state_dict(copy_state_dict(checkpoint['state_dict']))

    model.eval()
    return model


def get_prediction(image,
                   model=load_psenet_model(),
                   binary_th=1.0,
                   kernel_num=3,
                   upsample_scale=1,
                   long_size=1280,
                   min_kernel_area=10.0,
                   min_area=300.0,
                   min_score=0.93,
                   cuda=False):

    model = load_psenet_model(cuda=cuda,
                              scale=upsample_scale)

    scaled_img = scale_image(image, long_size)
    #scaled_img = np.expand_dims(scaled_img,axis=0)
    scaled_img = transforms.ToTensor()(scaled_img)
    scaled_img = transforms.Normalize(mean=[0.0618, 0.1206, 0.2677], std=[1.0214, 1.0212, 1.0242])(scaled_img)
    scaled_img = torch.unsqueeze(scaled_img, 0)
    #img = scaleimg(org_img)
    #img = img[:,:,[2,1,0]]
    #img = np.expand_dims(img,axis=0)
    #img = Image.fromarray(img)
    #img = img.convert('RGB')
    #img = torch.Tensor(img)
    #img = img.permute(0,3,1,2)

    outputs = model(scaled_img)

    score = torch.sigmoid(outputs[:, 0, :, :])
    outputs = (torch.sign(outputs - binary_th) + 1) / 2

    text = outputs[:, 0, :, :]
    kernels = outputs[:, 0:kernel_num, :, :] * text

    score = score.data.cpu().numpy()[0].astype(np.float32)
    text = text.data.cpu().numpy()[0].astype(np.uint8)
    kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)
    pred = pypse(kernels, min_kernel_area / (upsample_scale * upsample_scale))

    scale = (image.shape[1] * 1.0 / pred.shape[1], image.shape[0] * 1.0 / pred.shape[0])
    label = pred
    label_num = np.max(label) + 1
    boxes = []
    for i in range(1, label_num):
        points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

        if points.shape[0] < min_area / (upsample_scale * upsample_scale):
            continue

        score_i = np.mean(score[label == i])
        if score_i < min_score:
            continue

        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect) * scale
        box = box.astype('int32')
        boxes.append(box)
    boxes = np.array(boxes)

    # get image size
    img_height = image.shape[0]
    img_width = image.shape[1]

    # calculate box coords as ratios to image size
    boxes_as_ratio = []
    for box in boxes:
        boxes_as_ratio.append(box / [img_width, img_height])
    boxes_as_ratio = np.array(boxes_as_ratio)

    return {"boxes": boxes,
            "boxes_as_ratios": boxes_as_ratio}


if __name__ == '__main__':
    image_path = "figures/idcard.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction_result = get_prediction(image, long_size=1280)
