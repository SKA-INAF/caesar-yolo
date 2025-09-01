# caesar-yolo
Radio source detection with YOLO object detector

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add a reference to this github repository and acknowledge these works in your paper:   

* S. Riggi et al., *Astronomical source detection in radio continuum maps with deep neural networks*, 2023, Astronomy and Computing, 42, 100682, [doi](https://doi.org/10.1016/j.ascom.2022.100682)    

## **Installation**  

To build and install the package:    

* Download the software in a local directory, e.g. ```SRC_DIR```:   
  ```$ git clone https://github.com/SKA-INAF/caesar-yolo.git```   
* Create and activate a virtual environment, e.g. ```caesar-yolo```, under a desired path ```VENV_DIR```     
  ```$ python3 -m venv $VENV_DIR/caesar-yolo```    
  ```$ source $VENV_DIR/caesar-yolo/bin/activate```   
* Install dependencies inside venv:   
  ```(caesar-yolo)$ pip install -r $SRC_DIR/requirements.txt```   
* Build and install package in virtual env:   
  ```(caesar-yolo)$ python setup.py install```    
       
To use package scripts:

* Add binary directory to your ```PATH``` environment variable:   
  ``` export PATH=$PATH:$VENV_DIR/caesar-yolo/bin ```    

## **Usage**  

To detect source objects in input images use the the provided script ```run.py```:   

```(caesar-yolo)$ python $VENV_DIR/caesar-yolo/bin/run.py [OPTIONS]```    

Supported options are:  

**INPUT DATA**  
	`--image=[VALUE]`: Path to input image in FITS format   
 	`--datalist=[VALUE]`: Path to input data filelist containing a list of json files   
 	`--maxnimgs=[VALUE]`: Max number of images to consider in dataset (-1=all). Default: -1   
  	`--xmin=[VALUE]`: Image min x to be read (read all if -1). Default: -1   
  	`--xmax=[VALUE]`: Image max x to be read (read all if -1). Default: -1   
  	`--ymin=[VALUE]`: Image min y to be read (read all if -1). Default: -1   
  	`--ymax=[VALUE]`: Image max y to be read (read all if -1). Default: -1   	
  
**MODEL**  
	`--weights=[VALUE]`: Path to model weight file (.pt). This option is **mandatory**. Various pre-trained models are provided (see below).       
 
**DATA PRE-PROCESSING**     
  `--preprocessing`: Enable image pre-processing. Default: disabled   
  `--imgsize=[SIZE]`: Size in pixel used to resize input image. Default: 640     
  `--normalize_minmax`: Normalize each channel in range [norm_min, norm_max]. Default: no normalization    
  `--norm_min=[VALUE]`: Normalization min value. Default: 0.0    
  `--norm_max=[VALUE]`: Normalization max value. Default: 1.0   
  `--subtract_bkg`: Subtract bkg from ref channel image. Default: no subtraction  
  `--sigma_bkg=[VALUE]`: Sigma clip value used in bkg calculation. Default: 3.0  
  `--use_box_mask_in_bkg`: Compute bkg value in borders left from box mask. Default: not used   
  `--bkg_box_mask_fract=[VALUE]`: Size of mask box dimensions with respect to image size used in bkg calculation. Default: 0.7   
  `--bkg_chid=[VALUE]`: Channel used to subtract background (-1=all). Default: -1   
  `--clip_shift_data`: Apply sigma clip shifting. Default: not applied   
  `--sigma_clip=[VALUE]`: Sigma threshold to be used for clip & shifting pixels. Default: 1.0    
  `--clip_data`: Apply sigma clipping. Default: not applied    
  `--sigma_clip_low=[VALUE]`: Lower sigma threshold to be used for clipping pixels below (mean - sigma_low x stddev). Default: 10.0   
  `--sigma_clip_up=[VALUE]`: Upper sigma threshold to be used for clipping pixels above (mean + sigma_up x stddev). Default: 10.0     
  `--clip_chid=[VALUE]`: Channel used to clip data (-1=all). Default: -1     
  `--zscale_stretch`: Apply zscale transform to data. Default: not applied    
  `--zscale_contrasts=[VALUES]`: zscale contrasts applied to all channels, separated by commas. Default: 0.25,0.25,0.25        
  `--chan3_preproc`: Use the 3-channel pre-processor. Default: not used          
  `--sigma_clip_baseline=[VALUE]`: Lower sigma threshold to be used for clipping pixels below (mean - sigma_low x stddev) in first channel of 3-channel preprocessing. Default: 0.0         
  `--nchannels=[VALUE]`: Number of channels. If you modify channels in preprocessing you must set this option accordingly. Default: 1        

**SOURCE DETECTION**    
	`--scoreThr=[VALUE]`: Object detection score threshold, below which objects are not considered as sources. Default: 0.7       
	`--iouThr=[VALUE]`: Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS), below which objects are not considered as sources. Default: 0.5       
	`--merge_overlap_iou_thr_soft`: IOU soft threshold used to merge overlapping detected objects with same class. Default: 0.3       
	`--merge_overlap_iou_thr_hard`: IOU hard threshold used to merge overlapping detected objects even if from different classes. Default: 0.7       

**RUN**  
	`--devices=[VALUE]`: Specifies the device for inference (e.g., cpu, cuda:0 or 0). Default: cpu   
 	`--multigpu`: Enable multi-gpu inference. Default: disabled     

**PARALLEL PROCESSING**  
	`--split_img_in_tiles`: Enable splitting of input image in multiple subtiles for parallel processing. Default: disabled   
 	`--tile_xsize=[VALUE]`: Sub image size in pixel along x. Default: 512   
	`--tile_ysize=[VALUE]`: Sub image size in pixel along y. Default: 512   
	`--tile_xstep=[VALUE]`: Sub image step fraction along x (=1 means no overlap). Default: 1.0   
 	`--tile_ystep=[VALUE]`: Sub image step fraction along y (=1 means no overlap). Default: 1.0    
  	`--max_ntasks_per_worker=[VALUE]`: Max number of tasks assigned to a MPI processor worker. Default: 100      
   
**PLOTTING**  
	`--draw_plots`: Enable plotting of image and inference results superimposed. Default: disabled   
 	`--draw_class_label_in_caption`: Enable drawing of class labels inside detected source caption in inference plots. Default: disabled   

 **OUTPUT DATA**  
	`--save_plots`: Enable saving of inference plots. Default: disabled   
	`--save_tile_catalog`: Enable saving of catalog files for each subtile in parallel processing (debug scopes). Default: disabled   
 	`--save_tile_region`: Enable saving of DS9 region files for each subtile in parallel processing (debug scopes). Default: disabled   
  	`--save_tile_img`: Enable saving of subtile image in parallel processing (debug scopes). Default: disabled   
   	`--detect_outfile`: Output plot PNG filename (internally generated if left empty). Default: empty   
	`--detect_outfile_json`: Output json filename with detected objects (internally generated if left empty). Default: empty      	

Below, we report a sample run script:   

```
#!/bin/bash

# - Set env
VENV_DIR="/opt/software/venvs/caesar-yolo"
SCRIPT_DIR="$VENV_DIR/bin"
source $SCRIPT_DIR/activate

# - Set options
INPUTFILE="galaxy0001.fits"
WEIGHTFILE="weights-yolov8l_scratch_imgsize640_nepochs300.pt" # see pretrained weights below
PREPROC_OPTS="--preprocessing --imgsize=640 --zscale_stretch --zscale_contrasts=0.25,0.25,0.25 --normalize_minmax --norm_min=0 --norm_max=255 "
DET_OPTS="--scoreThr=0.5 --merge_overlap_iou_thr_soft=0.3 --merge_overlap_iou_thr_hard=0.8 "
DRAW_OPTS="--draw_plots --save_plots --draw_class_label_in_caption "

# - Run
python $SCRIPT_DIR/run.py --image=$INPUTFILE --weights=$WEIGHTFILE \
  	$PREPROC_OPTS \
  	$DET_OPTS \
  	$DRAW_OPTS \
	--devices="cuda:0"
```

Below, we report a sample parallel run script :   

```
#!/bin/bash

# - Set env
VENV_DIR="/opt/software/venvs/caesar-yolo"
SCRIPT_DIR="$VENV_DIR/bin"
source $SCRIPT_DIR/activate

# - Set options
INPUTFILE="G005.5+0.0IFx_Mosaic_Mom0.fits"
WEIGHTFILE="weights-yolov8l_scratch_imgsize512_nepochs300.pt" # see pretrained weights below
PREPROC_OPTS="--preprocessing --imgsize=512 --zscale_stretch --zscale_contrasts=0.25,0.25,0.25 --normalize_minmax --norm_min=0 --norm_max=255 "
DET_OPTS="--scoreThr=0.5 --merge_overlap_iou_thr_soft=0.3 --merge_overlap_iou_thr_hard=0.8 "
DRAW_OPTS="--draw_plots --save_plots --draw_class_label_in_caption "
PARALLEL_OPTS="--split_img_in_tiles --tile_xsize=512 --tile_ysize=512 --tile_xstep=1 --tile_ystep=1 "

# - Parallel run
mpirun -np 4 python $SCRIPT_DIR/run.py --image=$INPUTFILE --weights=$WEIGHTFILE \
  $PREPROC_OPTS \
  $DET_OPTS \
  $DRAW_OPTS \
  $PARALLEL_OPTS
```

## **Pre-trained models**  
We have trained various YOLO v8 models from scratch on the same annotated radio dataset that was previously used to train Mask R-CNN model in paper Riggi+2023 (see Credits for full reference). We provide below the training configuration and links to pre-trained model weights.

**Training configuration**
* epochs=300
* batch=16
* erasing=0,
* mosaic=0,
* hsv_h=0,
* hsv_s=0,
* hsv_v=0,
* translate=0,
* degrees=180,
* flipud=0.5,
* fliplr=0.5,
* scale=0.89
* crop_fraction=1.0

**Trained models**

| Model Base  | Img Size | Weights | File Size | Notes |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| yolov8n  | 640  | [url](https://tinyurl.com/2jznb7ra) | 6 MB | |
| yolov8l  | 128  | [url](https://tinyurl.com/53mm7svh) | 83.6 MB | |
| yolov8l  | 256  | [url](https://tinyurl.com/mr8yjumv) | 83.6 MB | |
| yolov8l  | 512  | [url](https://tinyurl.com/52p8bczz) | 83.6 MB | |
| yolov8l  | 512  | [url](https://tinyurl.com/2s4dvtdc) | 83.6 MB | scale=0.5 |
| yolov8l  | 640  | [url](https://tinyurl.com/4b5p4frc) | 83.6 MB | |
| yolov8l  | 1024  | [url](https://tinyurl.com/mry8c2b4) | 83.7 MB | |
