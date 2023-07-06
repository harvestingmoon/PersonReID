
# Person Re-Identification



Personal project for person reidentification. It uses YOLO-NAS for person detection while Centroid ReID for Person reidentification. The 2048 embeddings produced by Centroid ReID are then compared via Cosine Similarity. 

Currently, Centroid ReID achieves SOTA performance on the Market1501 benchmark.

YOLO-NAS also outperforms YOLO-V6 & V8 in terms of mAP.

## Video Feed

![Image_1](https://github.com/harvestingmoon/PersonReID/blob/main/images/reid.png)

Based on the image queries, you can place them in either blacklist or whitelist under data. Then, run ``` main.py ``` to run the program.

The script is designed to be multithreaded. I have also created a switch-key. Press ```a``` if you would like to disable ReID on the feed.

## How To Run 

1. ``` git pull https://github.com/harvestingmoon/PersonReID.git ``` 

2. ``` pip install -r requirements.txt ```

3. Configure the path to your image queries via ``` config.yaml ```

4. Download ``` market1501_resnet50_256_128_epoch_120.ckpt ``` and place it under ```/logs ``` as well as ``` resnet50-19c8e357.pth ``` and place it under ```models```

5. Place your blacklist and whitelist image queries under ``` /data ``` folder.

6. Run ``` main.py ```
There are mainly 3 files which I have created that made this possible
``` yolo_engine.py ``` , ``` reid_engine.py``` and ```main.py``` 

## Links to weights
ResNet-50: https://download.pytorch.org/models/resnet50-19c8e357.pth

Trained Model weights for CTL benchmark: https://drive.google.com/drive/folders/1NWD2Q0JGasGm9HTcOy4ZqsIqK4-IfknK

 




## Acknowledgements

Special thank you to the researchers for making the code open source.Below are the links to the original source code as well. 



YOLO-NAS/ SuperGradients : https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md


CTL/ Centroids-REID:  https://github.com/mikwieczorek/centroids-reid
