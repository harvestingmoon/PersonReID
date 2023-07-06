import argparse
import logging
import os
import sys
from pathlib import Path
import cv2
import cupy as cp 
import numpy as np
import torch
#from .reid_engine import cv2_converter 
from config import cfg
from train_ctl_model import CTLModel
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets.folder import is_image_file
from torchvision import transforms as T
from datasets.transforms import random_erasing
from PIL import Image
from yolo_engine import ImageBreakDown
# def pil_loader_dev(path: str) -> Image.Image:
#     with open(path,"rb") as f:
#         img = Image.open(f)
#         return img.convert("RGB")



def cv2_loader(path: str):
    image = cv2.imread(path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img

def cv2_converter(cv2_array):
    img = cv2.cvtColor(cv2_array, cv2.COLOR_BGR2RGB)
    return img

def pil_loader_dev(path: str) ->Image.Image:
    with open(path,"rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class ImageIndv:
    def __init__(self,image_dset,transform = None,loader = cv2_loader):
        self.image_dset= [image_dset]
       # print(self.image_dset)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.image_dset)
    
    def __getitem__(self,index):
        img_path = self.image_dset[index]
        img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
            img = img.to('cuda:0')
           # return img
        
        return (
            img,
            "",
            img_path
        )
    

class ImageIndv_dev:
    def __init__(self,image_array,image_name,transform = None):
        self.image_dset= [image_array]
        self.test_array = image_array
        self.image_name = image_name
        self.transform = transform
        #self.loader = loader

    def __len__(self):
        return len(self.image_dset)
    
        
    def __getitem__(self,index):
       # print(len(self.image_dset))
        try:
            img_opener = self.image_dset[index]
            img = cv2_converter(img_opener)
            name = self.image_name

            if self.transform is not None:
                img = self.transform(img)
                img = img.to('cuda:0')
            # return img
            
            return (
                img,
                "",
                name
            )
        except Exception as e:
            print(e)
            print(self.image_dset)


class ReidTransforms_Dev():

    def __init__(self, cfg):
        self.cfg = cfg

    def build_transforms(self, is_train=True):
        normalize_transform = T.Normalize(mean=self.cfg.INPUT.PIXEL_MEAN, std=self.cfg.INPUT.PIXEL_STD)
        if is_train:
            transform = T.Compose([
                T.ToTensor(),
                T.Resize(self.cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=self.cfg.INPUT.PROB),
                T.Pad(self.cfg.INPUT.PADDING),
                T.RandomCrop(self.cfg.INPUT.SIZE_TRAIN),
                normalize_transform,
                random_erasing.RandomErasing(probability=self.cfg.INPUT.RE_PROB, mean=self.cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Resize(self.cfg.INPUT.SIZE_TEST),
                normalize_transform
            ])

        return transform

class Combined_Indv:
    def __init__(self,ImageBreak,vals) -> None:
        self.det = ImageBreak
        self.val = vals
        
class ReID_Obj_Indv(Combined_Indv):
    def __init__(self, ImageBreak, vals,det) -> None:
        super().__init__(ImageBreak, vals)
        self.detected = det
    def change_det(self,det:bool):
        self.detected = det
    

def indv_image_transform(cfg,img,img_name,dclass):
    transforms_base = ReidTransforms_Dev(cfg)
    val_transforms = transforms_base.build_transforms(is_train=False) # strictly use it for inference
    # num_workers = cfg.DATALOADER.NUM_WORKERS
    val_set = dclass(img,img_name, transform =val_transforms)
    val_loader = DataLoader(
        val_set[0],
        batch_size = 1,
        shuffle = False,
        num_workers = 0 # must set this to 0 , not sure why if i readjust workers it will cause the error to appear
    )

    return val_loader


def _inference(model, batch, use_cuda=True, normalize_with_bn=True):
    model.cuda()
    model.eval()
    with torch.no_grad():
        data, _, filename = batch
        _, global_feat = model.backbone(
            data.cuda() if use_cuda else data
        )
        if normalize_with_bn:
            global_feat = model.bn(global_feat)
        return global_feat, filename

def comparison(preprocessed_list,detect_reid_list,threshold=0.65):
    for detect_obj in detect_reid_list:
        for preprocessed in preprocessed_list:
            detect_obj_a1 = detect_obj.val
            preprocessed_a1 = preprocessed.val
            preprocessed_name =  preprocessed.det.name
            cosine_vals = cosine_similarity(detect_obj_a1,preprocessed_a1)
          #  print(cosine_vals[0][0])
            if cosine_vals[0][0] > threshold:
                detect_obj.det.name = preprocessed_name
                detect_obj.det.type = preprocessed.det.type
                detect_obj.change_det(True) #so now when we asked for detected it returns us as True

    return detect_reid_list
                
def face_drawer(frame,filterd_list):
    for obj in filterd_list:
        bound_box = obj.det.bbox
        name = obj.det.name
        font = cv2.FONT_HERSHEY_COMPLEX
        x,y,w,h = int(bound_box[0]),int(bound_box[1]),int(bound_box[2]),int(bound_box[3])
        if obj.detected == True:
            if obj.det.type == "blacklist":
                cv2.rectangle(frame,(x,h),(w,y),(0,0,255),2)
                cv2.putText(frame,name,(x+1, h+1),font,1,(0,0,0),2)
            elif obj.det.type == "vip":
                cv2.rectangle(frame,(x,h),(w,y),(0,255,0),2)
                cv2.putText(frame,name,(x+1, h+1),font,1,(0,0,0),2)
        else:
            cv2.rectangle(frame,(x,h),(w,y),(0,0,0),3)
            cv2.putText(frame,name,(x, h+5),font,1,(0,0,0),2)
    return frame
            
#calculating the cosine similiarity
def cosine_similarity(x1,x2):
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    x2 = x2.T
    x1 = cp.asarray(x1)
    x2 = cp.asarray(x2)
    dot_product = cp.dot(x1,x2)
    norm_a = cp.linalg.norm(x1)
    norm_b = cp.linalg.norm(x2)
    
    return (dot_product/ (norm_a * norm_b))


