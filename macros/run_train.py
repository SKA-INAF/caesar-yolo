from ultralytics import YOLO
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


###########################
##    MODEL LOAD
##########################
# Load a model
model = YOLO("yolov8l.yaml")  # build a new model from scratch
###model = YOLO("runs/detect/train/weights/last.pt")  # build a new model from scratch

##########################
##  TRAIN MODEL
##########################
# Use the model
dataset= "dataset.yaml"

model.train(
  data=dataset,
  epochs=300,
  ##resume=True,
  ##batch=-1,
  batch=16,
  imgsz=640,
  device=[0,1,2,3],
  workers=4,
  pretrained=False,
  optimizer='auto',
  val=True,
  plots=True,
  ## AUGMENTATION OPTIONS
  erasing=0,
  mosaic=0,
  hsv_h=0,
  hsv_s=0,
  hsv_v=0,
  translate=0,
  degrees=180, #random rotation-180 180
  flipud=0.5, # random lr flipping
  fliplr=0.5, # random lr flipping
  scale=0.89, #scaling between 400x400, 850*850
  crop_fraction=1.0, #random cropping, bounds not specified in paper
)  # train the model

