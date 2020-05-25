import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from utils.vis import vis_keypoints
from utils.pose_utils import flip
import torch.backends.cudnn as cudnn
from utils.pose_utils import pixel2cam
import torchvision.transforms as transforms
from dataset import DatasetLoader
exec('from ' + cfg.testset + ' import ' + cfg.testset)

testset = eval(cfg.testset)("test")
testset_loader = DatasetLoader(testset, None, False, transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)]))
testset_loader[0]
