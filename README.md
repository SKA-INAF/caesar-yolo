# caesar-yolo
Radio source detection with YOLO object detector

## **Credit**
This software is distributed with GPLv3 license. If you use it for your research, please add repository link or acknowledge authors in your papers.   

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

*INPUT DATA*  
	`--image=[VALUE]`: Path to input image in FITS format   
 	`--datalist=[VALUE]`: Path to input data filelist containing a list of json files   
 	`--maxnimgs=[VALUE]`: Max number of images to consider in dataset (-1=all). Default: -1   
 
*MODEL*  
	`--weights=[VALUE]`: Path to model weight file (.pt). This option is **mandatory**    
 
*DATA PRE-PROCESSING*     
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
	
*RUN*  
	`--devices=[VALUE]`: Specifies the device for inference (e.g., cpu, cuda:0 or 0). Default: cpu   
 	`--multigpu`: Enable multi-gpu inference. Default: disabled      
 
