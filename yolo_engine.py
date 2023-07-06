
class ImageBreakDown:
    def __init__(self,name,person_type,bbox,image) -> None:
        self.name = name
        self.type = person_type
        self.bbox = bbox
        self.image = image
    
    def name_change(self,name):
        self.name = name
        return self.name

        

def yolo_detector(frame,yolo_model,conf = 0.25):
   # model = models.get(yolo_model, pretrained_weights= pretrained_weights).cuda() (call it once only via fusing model)
    prediction = yolo_model.predict(frame,conf=conf,fuse_model= False)
    prediction_objs = list(prediction._images_prediction_lst)[0]
    bboxes = prediction_objs.prediction.bboxes_xyxy
    int_labels = prediction_objs.prediction.labels.astype(int)
    class_names = prediction_objs.class_names
    pred_classes = [class_names[i] for i in int_labels]
    all_cropped_images = []
  #  print(pred_classes)
    for i,classes in enumerate(pred_classes):
        if classes == "person":
                person = bboxes[i]
                x,y,w,h = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                cropped_image = frame[y:h,x:w]
                bound_box = [x,y,w,h]
                all_cropped_images.append(ImageBreakDown("Person","Normal",bound_box,cropped_image))
        
   # print(len(all_cropped_images))
    return all_cropped_images

def preprocess_det(image,name,type,yolo_model,conf = 0.25):
     # model = models.get(yolo_model, pretrained_weights= pretrained_weights).cuda() (call it once only via fusing model)
    prediction = yolo_model.predict(image,conf=conf,fuse_model= False)
    prediction_objs = list(prediction._images_prediction_lst)[0]
    bboxes = prediction_objs.prediction.bboxes_xyxy
    int_labels = prediction_objs.prediction.labels.astype(int)
    class_names = prediction_objs.class_names
    pred_classes = [class_names[i] for i in int_labels]
    all_cropped_images = []
  #  print(pred_classes)
    for i,classes in enumerate(pred_classes):
        if classes == "person":
                person = bboxes[i]
                x,y,w,h = int(person[0]),int(person[1]),int(person[2]),int(person[3])
                cropped_image = image[y:h,x:w]
                bound_box = [x,y,w,h]
                if type == "blacklist":
                    all_cropped_images.append(ImageBreakDown(name,type,bound_box,cropped_image))
                
                elif type == "vip":
                    all_cropped_images.append(ImageBreakDown(name,type,bound_box,cropped_image))
                
    
    return all_cropped_images

