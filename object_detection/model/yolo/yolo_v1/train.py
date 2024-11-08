#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 20:34:04 2024

@author: yeahia
"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from model import YoloV1
from loss import YoloLoss
from dataset import VOCDataset
from utils import load_checkpoint


seed = 123
torch.manual_seed(seed)

EPOCH = 1000
LEARNING_RATE = 2e-5
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
WEIGHT_DECAY = 0
NUM_WORKERS = 2
LOAD_MODEL = False
PIN_MEMORY = True
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "/kaggle/input/pascalvoc-yolo/images"
LABEL_DIR = "/kaggle/input/pascalvoc-yolo/labels"
CSV_DIR_TRAIN = "/kaggle/input/pascalvoc-yolo/100examples.csv"
CSV_DIR_TEST = "/kaggle/input/pascalvoc-yolo/test.csv"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        
        return img, bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    print('hello')
    
def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    
    loss_fn = YoloLoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    
    train_dataset = VOCDataset(
        CSV_DIR_TRAIN, 
        transform=transform,
        img_dir = IMG_DIR, 
        label_dir= LABEL_DIR
    )
    test_dataset = VOCDataset(
        CSV_DIR_TEST, 
        transform=transform, 
        img_dir=IMG_DIR, 
        label_dir=LABEL_DIR,
    )
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
        shuffle = True,
        drop_last = True
    )
    # for epoch in range(EPOCH):
    #     pred_boxes, target_boxes = get_bboxes(
    #         train_loader,
    #         model,
    #         iou_threshold = 0.5,
    #         threshold = 0.4,
    #     )
        
    
    
if __name__ == "__main__":
    main()