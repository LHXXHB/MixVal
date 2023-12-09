import numpy as np
from torchvision import transforms
import os
import random
from PIL import Image, ImageOps, ImageFilter
import numbers
import torch

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
      ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
 
  return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize
      ])


def image_visda(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
      ])