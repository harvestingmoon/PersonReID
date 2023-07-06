import cv2
import yaml
import os
from super_gradients.training import models
from pathlib import Path
from train_ctl_model import CTLModel
import yolo_engine
from reid_engine import ImageIndv_dev,Combined_Indv,ReID_Obj_Indv
import reid_engine
from config import cfg


with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

def prep_yolo_model(yolo: str, pretrained_weights ="coco"):
    model = models.get(yolo,pretrained_weights = pretrained_weights).cuda()
    return model

def prep_reid(path_to_checkpt):
    model = CTLModel.load_from_checkpoint(path_to_checkpt)
    return model

#preprocessed is done but need to slowly start testing this soon.. afterwards we will do the main folder ah
def preprocess(type,yolo_model,reid_model,file):
    directory = file
    preprocessed_list = []
    for fname in sorted(os.listdir(directory)):
        path = os.path.join(directory,fname)
        img_read = cv2.imread(path)
        indv_name = Path(path).stem
        cropped_image = yolo_engine.preprocess_det(img_read,indv_name,type,yolo_model)
        image_indv = cropped_image[0].image
        indv_name = cropped_image[0].name
        val_loader = reid_engine.indv_image_transform(cfg,image_indv,indv_name,ImageIndv_dev)
        a1,name = reid_engine._inference(reid_model,val_loader) # name is basically the same as what is iside Imagebreakdown
        preprocessed_list.append(Combined_Indv(cropped_image[0],a1))
        
    return preprocessed_list



def video(yolo,reid):
    video_feed = cv2.VideoCapture(config['video_feed'])
    yolo_model = prep_yolo_model(yolo)
    reid_model = prep_reid(reid)
    blacklist_list = preprocess("blacklist",yolo_model,reid_model,file= config['blacklist'])
    whitelist_list = preprocess("vip",yolo_model,reid_model,file= config['whitelist'])
    preprocessed_list = blacklist_list + whitelist_list
    while True:
        ret,frame = video_feed.read()
        cropped_images = yolo_engine.yolo_detector(frame,yolo_model)
        detect_reid_list = []
        for cropped_obj in cropped_images:
            cropped_img = cropped_obj.image
            indv_name = cropped_obj.name
            if cropped_img.size == 0:
                continue
            val_loader = reid_engine.indv_image_transform(cfg,cropped_img,indv_name,ImageIndv_dev)
            a1,name = reid_engine._inference(reid_model,val_loader)
            detect_reid_list.append(ReID_Obj_Indv(cropped_obj,a1,False)) # this is the assumption that the name is just normal ( we do not know the actual name yet)
        filtered_list = reid_engine.comparison(preprocessed_list,detect_reid_list,threshold= config['threshold'])
        drawn_frame = reid_engine.face_drawer(frame,filtered_list)
        cv2.imshow("ReID_frame",drawn_frame)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            break
    video_feed.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video(config['yolo_version'],config['reid_checkpt'])