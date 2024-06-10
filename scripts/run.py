#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import json
import time
import argparse
import datetime
import random
import numpy as np
import copy

## MODULES
from ultralytics import YOLO

## USER MODULES
from caesar_yolo import logger
from caesar_yolo.config import CONFIG

#from caesar_yolo.dataset import Dataset
from caesar_yolo.preprocessing import DataPreprocessor
from caesar_yolo.preprocessing import BkgSubtractor, SigmaClipper, SigmaClipShifter, Scaler, LogStretcher
from caesar_yolo.preprocessing import Resizer, MinMaxNormalizer, AbsMinMaxNormalizer, MaxScaler, AbsMaxScaler, ChanMaxScaler
from caesar_yolo.preprocessing import Shifter, Standardizer, ChanDivider, BorderMasker
from caesar_yolo.preprocessing import ChanResizer, ZScaleTransformer, Chan3Trasformer
from caesar_yolo.inference import SFinder

#===========================
#==   IMPORT MPI
#===========================
MPI= None
comm= None
nproc= 1
procId= 0

############################################################
#        PARSE/VALIDATE ARGS
############################################################

def parse_args():
	""" Parse command line arguments """  
  
	# - Parse command line arguments
	parser = argparse.ArgumentParser(description='CAESAR-YOLO options')

	# - DATA PRE-PROCESSING OPTIONS
	parser.add_argument('--imgsize', dest='imgsize', required=False, type=int, default=256, help='Size in pixel used to resize input image (default=256)')
	
	parser.add_argument('--normalize_minmax', dest='normalize_minmax', action='store_true',help='Normalize each channel in range. Default: [0,1]')	
	parser.set_defaults(normalize_minmax=False)
	parser.add_argument('-norm_min', '--norm_min', dest='norm_min', required=False, type=float, default=0., action='store',help='Normalization min value (default=0)')
	parser.add_argument('-norm_max', '--norm_max', dest='norm_max', required=False, type=float, default=1., action='store',help='Normalization max value (default=1)')
	
	parser.add_argument('--subtract_bkg', dest='subtract_bkg', action='store_true',help='Subtract bkg from ref channel image')	
	parser.set_defaults(subtract_bkg=False)
	parser.add_argument('-sigma_bkg', '--sigma_bkg', dest='sigma_bkg', required=False, type=float, default=3, action='store',help='Sigma clip to be used in bkg calculation (default=3)')
	parser.add_argument('--use_box_mask_in_bkg', dest='use_box_mask_in_bkg', action='store_true',help='Compute bkg value in borders left from box mask')	
	parser.set_defaults(use_box_mask_in_bkg=False)	
	parser.add_argument('-bkg_box_mask_fract', '--bkg_box_mask_fract', dest='bkg_box_mask_fract', required=False, type=float, default=0.7, action='store',help='Size of mask box dimensions with respect to image size used in bkg calculation (default=0.7)')
	parser.add_argument('-bkg_chid', '--bkg_chid', dest='bkg_chid', required=False, type=int, default=-1, action='store',help='Channel to subtract background (-1=all) (default=-1)')

	parser.add_argument('--clip_shift_data', dest='clip_shift_data', action='store_true',help='Do sigma clipp shifting')	
	parser.set_defaults(clip_shift_data=False)
	parser.add_argument('-sigma_clip', '--sigma_clip', dest='sigma_clip', required=False, type=float, default=1, action='store',help='Sigma threshold to be used for clip & shifting pixels (default=1)')
	parser.add_argument('--clip_data', dest='clip_data', action='store_true',help='Do sigma clipping')	
	parser.set_defaults(clip_data=False)
	parser.add_argument('-sigma_clip_low', '--sigma_clip_low', dest='sigma_clip_low', required=False, type=float, default=10, action='store',help='Lower sigma threshold to be used for clipping pixels below (mean-sigma_low*stddev) (default=10)')
	parser.add_argument('-sigma_clip_up', '--sigma_clip_up', dest='sigma_clip_up', required=False, type=float, default=10, action='store',help='Upper sigma threshold to be used for clipping pixels above (mean+sigma_up*stddev) (default=10)')	
	parser.add_argument('-clip_chid', '--clip_chid', dest='clip_chid', required=False, type=int, default=-1, action='store',help='Channel to clip data (-1=all) (default=-1)')
	
	parser.add_argument('--zscale_stretch', dest='zscale_stretch', action='store_true',help='Do zscale transform')	
	parser.set_defaults(zscale_stretch=False)
	parser.add_argument('--zscale_contrasts', dest='zscale_contrasts', required=False, type=str, default='0.25,0.25,0.25',help='zscale contrasts applied to all channels') 
	
	parser.add_argument('--chan3_preproc', dest='chan3_preproc', action='store_true',help='Use the 3 channel pre-processor')	
	parser.set_defaults(chan3_preproc=False)
	parser.add_argument('-sigma_clip_baseline', '--sigma_clip_baseline', dest='sigma_clip_baseline', required=False, type=float, default=0, action='store',help='Lower sigma threshold to be used for clipping pixels below (mean-sigma_low*stddev) in first channel of 3-channel preprocessing (default=0)')
	parser.add_argument('-nchannels', '--nchannels', dest='nchannels', required=False, type=int, default=1, action='store',help='Number of channels (1=default). If you modify channels in preprocessing you must set this accordingly')
	
	# - DATA OPTIONS
	parser.add_argument('--image', required=False, metavar="Input image", type=str, help='Input image in FITS format to apply the model (used in detect task)')
	parser.add_argument('--datalist', required=False, metavar="/path/to/dataset", help='Train/test data filelist containing a list of json files')
	parser.add_argument('--maxnimgs', required=False, metavar="", type=int, default=-1, help="Max number of images to consider in dataset (-1=all) (default=-1)")

	# - MODEL OPTIONS
	parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5", help="Path to weights .h5 file")
	
	# - DETECT OPTIONS
	parser.add_argument('--scoreThr', required=False, default=0.7, type=float, metavar="Object detection score threshold to be used during test",help="Object detection score threshold to be used during test")
	#parser.add_argument('--iouThr', required=False, default=0.6, type=float, metavar="IOU threshold used to match detected objects with true objects",help="IOU threshold used to match detected objects with true objects")
	parser.add_argument('--merge_overlap_iou_thr_soft', required=False, default=0.3, type=float, metavar="IOU threshold used to merge overlapping detected objects with same class",help="IOU threshold used to merge overlapping detected objects with same class")
	parser.add_argument('--merge_overlap_iou_thr_hard', required=False, default=0.8, type=float, metavar="IOU threshold used to merge overlapping detected objects",help="IOU threshold used to merge overlapping detected objects")
	
	parser.add_argument('--xmin', dest='xmin', required=False, type=int, default=-1, help='Image min x to be read (read all if -1)') 
	parser.add_argument('--xmax', dest='xmax', required=False, type=int, default=-1, help='Image max x to be read (read all if -1)') 
	parser.add_argument('--ymin', dest='ymin', required=False, type=int, default=-1, help='Image min y to be read (read all if -1)') 
	parser.add_argument('--ymax', dest='ymax', required=False, type=int, default=-1, help='Image max y to be read (read all if -1)') 
	parser.add_argument('--detect_outfile', required=False, metavar="Output plot filename", type=str, default="", help='Output plot PNG filename (internally generated if left empty)')
	parser.add_argument('--detect_outfile_json', required=False, metavar="Output json filename with detected objects", type=str, default="", help='Output json filename with detected objects (internally generated if left empty)')

	# - PARALLEL PROCESSING OPTIONS
	parser.add_argument('--split_img_in_tiles', dest='split_img_in_tiles', action='store_true')	
	parser.set_defaults(split_img_in_tiles=False)
	parser.add_argument('--tile_xsize', dest='tile_xsize', required=False, type=int, default=512, help='Sub image size in pixel along x') 
	parser.add_argument('--tile_ysize', dest='tile_ysize', required=False, type=int, default=512, help='Sub image size in pixel along y') 
	parser.add_argument('--tile_xstep', dest='tile_xstep', required=False, type=float, default=1.0, help='Sub image step fraction along x (=1 means no overlap)') 
	parser.add_argument('--tile_ystep', dest='tile_ystep', required=False, type=float, default=1.0, help='Sub image step fraction along y (=1 means no overlap)') 

	# - DRAW OPTIONS
	parser.add_argument('--draw_plots', dest='draw_plots', action='store_true')	
	parser.set_defaults(draw_plots=False)
	parser.add_argument('--draw_class_label_in_caption', dest='draw_class_label_in_caption', action='store_true')	
	parser.set_defaults(draw_class_label_in_caption=False)
	parser.add_argument('--save_plots', dest='save_plots', action='store_true')	
	parser.set_defaults(save_plots=False)
	
	args = parser.parse_args()

	return args


def validate_args(args):
	""" Validate arguments """
	
	# - Check image arg
	has_image= (args.image and args.image!="")
	image_exists= os.path.isfile(args.image)
	valid_extension= args.image.endswith('.fits')
	if not has_image:
		logger.error("Argument --image is required for detect task!")
		return -1
	if not image_exists:
		logger.error("Image argument must be an existing image on filesystem!")
		return -1
	if not valid_extension:
		logger.error("Image must have .fits extension!")
		return -1

	# - Check maxnimgs
	if args.maxnimgs==0 or (args.maxnimgs<0 and args.maxnimgs!=-1):
		logger.error("Invalid maxnimgs given (hint: give -1 or >0)!")
		return -1

	# - Check weight file exists
	if args.weights=="":
		logger.error("Empty weight file path!")
		return -1
	else:
		check= os.path.exists(args.weights) and os.path.isfile(args.weights)
		if not check:
			logger.error("Given weight file %s not existing or not a file!" % (args.weights))
			return -1
		
	# - Check remap id
	if args.remap_classids:
		if args.classid_remap_dict=="":
			logger.error("Classid remap dictionary is empty (you need to provide one if you give the option --remap_classids)!")
			return -1

	return 0
	
	

	
############################################################
#        DETECT
############################################################
def run_inference(args, model, config):
	""" Test the model on input dataset with ground truth knowledge """ 

	# - Create sfinder and detect sources
	sfinder= SFinder(model, config)

	if args.split_img_in_tiles:
		logger.info("Running sfinder parallel version ...")
		status= sfinder.run_parallel()
	else:
		logger.info("Running sfinder serial version ...")
		status= sfinder.run()

	if status<0:
		logger.error("sfinder run failed, see logs...")
		return -1

	return 0

############################################################
#       MAIN
############################################################
def main():
	"""Main function"""

	#===========================
	#==   PARSE ARGS
	#===========================
	if procId==0:
		logger.info("[PROC %d] Parsing script args ..." % procId)
	try:
		args= parse_args()
	except Exception as ex:
		logger.error("[PROC %d] Failed to get and parse options (err=%s)" % (procId, str(ex)))
		return 1

	#===========================
	#==   VALIDATE ARGS
	#===========================
	if procId==0:
		logger.info("[PROC %d] Validating script args ..." % procId)
	if validate_args(args)<0:
		logger.error("[PROC %d] Argument validation failed, exit ..." % procId)
		return 1

	if procId==0:
		print("Weights: ", args.weights)
		print("scoreThr: ",args.scoreThr)
		print("classdict: ",args.classdict)

	#===========================
	#==   SET PARAMETERS
	#===========================
	# - Set data pre-processing options
	zscale_contrasts= [float(x) for x in args.zscale_contrasts.split(',')]
	if args.chan3_preproc and args.nchannels!=3:
		logger.error("You selected chan3_preproc pre-processing options, you must set nchannels options to 3!")
		return 1
	
	# - Set model options
	weights_path= None
	if args.weights!="":
		weights_path= args.weights
		
	#==============================
	#==   DEFINE PRE-PROCESSOR
	#==============================
	preprocess_stages= []

	if args.subtract_bkg:
		preprocess_stages.append(BkgSubtractor(sigma=args.sigma_bkg, use_mask_box=args.use_box_mask_in_bkg, mask_fract=args.bkg_box_mask_fract, chid=args.bkg_chid))

	if args.clip_shift_data:
		preprocess_stages.append(SigmaClipShifter(sigma=args.sigma_clip, chid=args.clip_chid))

	if args.clip_data:
		preprocess_stages.append(SigmaClipper(sigma_low=args.sigma_clip_low, sigma_up=args.sigma_clip_up, chid=args.clip_chid))

	if args.nchannels>1:
		preprocess_stages.append(ChanResizer(nchans=args.nchannels))

	if args.zscale_stretch:
		preprocess_stages.append(ZScaleTransformer(contrasts=zscale_contrasts))

	if args.chan3_preproc:
		preprocess_stages.append( Chan3Trasformer(sigma_clip_baseline=args.sigma_clip_baseline, sigma_clip_low=args.sigma_clip_low, sigma_clip_up=args.sigma_clip_up, zscale_contrast=zscale_contrasts[0]) )

	if args.normalize_minmax:
		preprocess_stages.append(MinMaxNormalizer(norm_min=args.norm_min, norm_max=args.norm_max))

	logger.info("[PROC %d] Data pre-processing steps: %s" % (procId, str(preprocess_stages)))
	
	dp= None
	if not preprocess_stages:
		logger.warn("No pre-processing steps defined ...")
	else:
		dp= DataPreprocessor(preprocess_stages)

	#===========================
	#==   CONFIG
	#===========================
	# - Override some other options
	logger.info("Setting config options ...")
	
		
	# - Set detection options
	CONFIG['img_size']= args.imgsize
	CONFIG['preprocess_fcn']= dp
	CONFIG['image_path']= args.image
	CONFIG['image_xmin']= args.xmin
	CONFIG['image_xmax']= args.xmax
	CONFIG['image_ymin']= args.ymin
	CONFIG['image_ymax']= args.ymax
	CONFIG['mpi']= MPI
	CONFIG['split_image_in_tiles']= args.split_img_in_tiles
	CONFIG['tile_xsize']= args.tile_xsize
	CONFIG['tile_ysize']= args.tile_ysize
	CONFIG['tile_xstep']= args.tile_xstep
	CONFIG['tile_ystep']= args.tile_ystep
	#CONFIG['iou_thr']= args.iouThr
	CONFIG['score_thr']= args.scoreThr
	CONFIG['merge_overlap_iou_thr_soft']= args.merge_overlap_iou_thr_soft
	CONFIG['merge_overlap_iou_thr_hard']= args.merge_overlap_iou_thr_hard	
	CONFIG['outfile']= args.detect_outfile
	CONFIG['outfile_json']= args.detect_outfile_json
	CONFIG['draw_plot']= args.draw_plots
	CONFIG['draw_class_label_in_caption']= args.draw_class_label_in_caption
	CONFIG['save_plot']= args.save_plots

	logger.info("[PROC %d] Config options: %s" % (procId, str(CONFIG)))

	#===========================
	#==   CREATE MODEL
	#===========================
	# - Creating the model
	logger.info("[PROC %d] Creating YOLO model, loading weights from file %s ..." % (procId, weights_path))
	model = YOLO(model_weights)
		
	#===========================
	#==   RUN
	#===========================
	if run_inference(args, model=model, config=CONFIG)<0:
		logger.error("[PROC %d] Failed to run model inference!" % procId)
		return 1
	
	return 0


###################
##   MAIN EXEC   ##
###################
if __name__ == "__main__":
	sys.exit(main())
