# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import importlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# model_dir='micalib.models', model_name='mica'
def find_model_using_name(model_dir, model_name):
    # adapted from pix2pix framework: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/__init__.py#L25
    # import "model_dir/modelname.py"
    model_filename = model_dir + "." + model_name
    # importlib에서 model_filename과 model_dir에 따라 모듈을 불러온다.(import 한다. importlib은 import를 수행하는 함수)
    # 이렇게 import를 함수까지 써가며 복잡하게 모듈을 불러오는 이유는 우리가 알고 있는 방식으로 import하면 모듈의 값이 바뀌어도
    # 변경된 내용이 저장되지 않기 때문이다.
    modellib = importlib.import_module(model_filename, package=model_dir)

    # In the file, the class called ModelName() will
    # be instantiated. It has to be a subclass of BaseModel,
    # and it is case-insensitive.
    model = None
    target_model_name = model_name.replace('_', '')

    # 모델을 할당한다. modellib의 인스턴스(mica.py를 import한 것)의
    # 변수들을 __dict__로 불러와서(__dict__는 인스턴스 변수들을 dict형태로 만들어 출력한다.)
    # 원하는 모델의 이름(target_model_name)과 일치하는 변수(모델, name)이 있다면
    # model 변수에 할당한다.
    # 참고로 기본 모델로서 제공되는 모델의 이름은 'mica'
    for name, cls in modellib.__dict__.items():
        # if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a class with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def visualize_grid(visdict, savepath=None, size=224, dim=1, return_gird=True):
    '''
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    '''
    assert dim == 1 or dim == 2
    grids = {}
    for key in visdict:
        b, c, h, w = visdict[key].shape
        if dim == 2:
            new_h = size
            new_w = int(w * size / h)
        elif dim == 1:
            new_h = int(h * size / w)
            new_w = size
        grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu(), nrow=b, padding=0)
    grid = torch.cat(list(grids.values()), dim)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    if savepath:
        cv2.imwrite(savepath, grid_image)
    if return_gird:
        return grid_image
