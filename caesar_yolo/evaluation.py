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
import math
import datetime
import collections
import csv
import logging
from typing import List		# for type annotation
import numpy as np

## Import graphics modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches, lines

## Import regions module
import regions
from regions import RectanglePixelRegion, PixCoord

## USER MODULES
from caesar_yolo import logger
from caesar_yolo.graph import Graph
from caesar_yolo import utils

# ========================
# ==    ANALYZER
# ========================
class Analyzer(object):
	""" Define analyzer object """

	def __init__(self, model, config):
		""" Return an analyzer object """

		# - Model
		self.model= model
		self.class_names= self.model.names
		self.n_classes= len(self.model.names)	
		self.r= None

		# - Config options
		self.config= config
		
		# - Data options
		self.image= None
		self.image_header= None
		self.image_id= -1
		self.image_uuid= ''
		self.image_path= ''
		self.image_path_base= ''
		self.image_path_base_noext= ''
		self.image_xmin= 0
		self.image_ymin= 0
		
		# - Model inference data (before bbox merging)
		self.bboxes= None
		self.class_ids= None
		self.scores= None
		self.labels= None
		
		# - Model inference data (after bbox merging)
		self.bboxes_final= []
		self.class_ids_final= []
		self.scores_final= []
		self.labels_final= []
		
		self.results= {}     # dictionary with detected objects
		self.obj_name_tag= ""
		self.obj_regions= [] # list of DS9 region objects

		# - Detection process options
		self.imgsize= config['img_size']
		self.device= config['devices'][0]
		self.iou_thr= config['iou_thr']
		self.score_thr= config['score_thr']
		self.merge_overlap_iou_thr_soft= config['merge_overlap_iou_thr_soft']
		self.merge_overlap_iou_thr_hard= config['merge_overlap_iou_thr_hard']

		# - Draw options
		self.outfile= ""
		self.outfile_json= ""
		self.outfile_ds9= ""
		self.outfile_img= ""
		self.save_plots= config['save_plot']
		self.draw= config['draw_plot']
		self.draw_class_label_in_caption= config['draw_class_label_in_caption']
		self.write_to_json= config['save_catalog']
		self.write_to_ds9= config['save_region']
		self.save_img= config['save_img']
		
		self.class_color_map= {
			'bkg': (0,0,0),# black
			'spurious': (1,0,0),# red
			'compact': (0,0,1),# blue
			'extended': (1,1,0),# green	
			'extended-multisland': (1,0.647,0),# orange
			'flagged': (0,0,0),# black
		}
		self.class_color_map_ds9= {
			'bkg': "black",
			'spurious': "red",
			'compact': "blue",
			'extended': "green",	
			'extended-multisland': "orange",
			'flagged': "magenta",
		}


	def set_image_path(self, path):
		""" Set image path """
		self.image_path= path
		self.image_path_base= os.path.basename(self.image_path)
		self.image_path_base_noext= os.path.splitext(self.image_path_base)[0]


	# ========================
	# ==     PREDICT
	# ========================
	def predict(self, image, image_id='', header=None, xmin=0, ymin=0):
		""" Predict results on given image """
		
		# - Throw error if image is None
		if image is None:
			logger.error("No input image given!")
			return -1
			
		self.image= image
		self.image_xmin= xmin
		self.image_ymin= ymin

		if image_id:
			self.image_id= image_id
		if header:
			self.image_header= header
			
		# - Convert to 3 channel format
		nchans= self.image.ndim
		img_shape= self.image.shape
		if nchans!=3:
			logger.info("Converting image (nchan=%d) to 3 channels ..." % (nchans))
			image_cube= np.zeros((img_shape[0], img_shape[1], 3))
			image_cube[:,:,0]= self.image
			image_cube[:,:,1]= self.image
			image_cube[:,:,2]= self.image
			self.image= image_cube
			
		# - Pre-process image?
		dp= self.config['preprocess_fcn']
		if dp is not None:
			logger.info("Apply pre-processing to input image ...")
			image_proc= dp(self.image)
			self.image= image_proc
			
		# - Check input image (None, channels with equal values)
		if self.image is None:
			logger.warn("Input image is None, no prediction made.")
			return -1
		
		nchans= self.image.ndim
		img_shape= self.image.shape
		
		for i in range(img_shape[-1]):
			img_min= np.min(self.image[i])
			img_max= np.max(self.image[i])
			if img_min==img_max:
				logger.warn("Input image (ch %d) pixels have the same value (%f), no prediction made." % (i+1, img_max))
				return -1
		
		# - Compute model predictions
		logger.info("Computing model prediction (imgsz=%d, iou=%f, conf=%f) ..." % (self.imgsize, self.iou_thr, self.score_thr))
		try:
			results= self.model(
				self.image, 
				save=False,
				device=self.device,
				imgsz=self.imgsize,
				conf=self.score_thr, 
				iou=self.iou_thr,
				visualize=False,
				show=False,
				show_labels=False,
				show_conf=False,
				show_boxes=False
			)
		except Exception as e:
			logger.warn("Model prediction failed (err=%s)..." % (str(e)))
			return -1
		
		# - Process predictions
		if self.process_detections(results)<0:
			logger.error("Failed to process model predictions!")
			return -1
			
		# - Draw results
		if self.draw:
			logger.info("Drawing results for image %s ..." % (str(self.image_id)))
			if self.outfile=="":
				outfile= 'out_' + str(self.image_id) + '.png'
			else:
				outfile= self.outfile
			self.draw_results(outfile)

		# - Create dictionary with detected objects
		self.make_json_results()

		# - Write json results?
		if self.write_to_json:
			logger.info("Writing results for image %s to json ..." % str(self.image_id))
			if self.outfile_json=="":
				outfile_json= 'out_' + str(self.image_id) + '.json'
			else:
				outfile_json= self.outfile_json
			self.write_json_results(outfile_json)

		# - Create DS9 region objects
		self.make_ds9_regions()

		# - Write DS9 regions to file?
		if self.write_to_ds9:
			logger.info("Writing detected objects for image %s to DS9 format ..." % str(self.image_id))
			if self.outfile_ds9=="":
				outfile_ds9= 'out_' + str(self.image_id) + '.reg'
			else:
				outfile_ds9= self.outfile_ds9
			self.write_ds9_regions(outfile_ds9)
			
		# - Write image in FITS format?
		if self.save_img:
			logger.info("Saving image %s in FITS format ..." % str(self.image_id))
			if self.outfile_img=="":
				outfile_img= 'out_' + str(self.image_id) + '.fits'
			else:
				outfile_img= self.outfile_img
			self.write_fits(outfile_img)
				
		return 0	
			
	
	
	# ===========================
	# ==  PROCESS DETECTIONS
	# ===========================
	def process_detections(self, results):
		""" Process model predictions """
	
		# - Loop through predictions and select bboxes > scores
		bboxes_det= []
		scores_det= []
		labels_det= []
		class_ids_det= []
  
		for result in results:
			bboxes= result.boxes.xyxy.cpu().numpy()   # box with xywh format, (N, 4)
			scores= result.boxes.conf.cpu().numpy()   # confidence score, (N, 1)
			cls= result.boxes.cls.cpu().numpy()    # cls, (N, 1)
			class_labels= [self.class_names[int(item)] for item in cls]
 	 	
			print("--> PREDICTION")
			print(bboxes)
			print("scores")
			print(scores)
			print("cls")
			print(cls)
			print(class_labels)
  
  		# - Select score > thr
			for i in range(len(scores)):
				score= scores[i]
				bbox= bboxes[i]
				label= class_labels[i]
				class_id= int(cls[i])
				
				if score<self.score_thr:
					continue
				scores_det.append(score)
				bboxes_det.append(bbox)
				labels_det.append(label)
				class_ids_det.append(class_id)
				
		self.bboxes= bboxes_det
		self.scores= scores_det
		self.class_ids= class_ids_det
		self.labels= labels_det
  		
		# - Find overlapped bboxes with same labels, keep only that with higher confidence
		N= len(bboxes_det)
		g= Graph(N)
		for i in range(N-1):
		
			for j in range(i+1,N):
				same_class= (labels_det[i]==labels_det[j])
			
				iou= utils.get_iou(bboxes_det[i], bboxes_det[j])
				overlapping_soft= (iou>=self.merge_overlap_iou_thr_soft)
				overlapping_hard= (iou>=self.merge_overlap_iou_thr_hard)
				mergeable= (overlapping_hard or (same_class and overlapping_soft))
				logger.info("IoU(%d,%d)=%f, mergeable? %d" % (i+1, j+1, iou, mergeable))
  
				if mergeable:
					g.addEdge(i,j)
  
  	# - Select connected boxes
		cc = g.connectedComponents()
		bboxes_sel= []
		scores_sel= []
		labels_sel= []
		class_ids_sel= []
  
		for i in range(len(cc)):
			if not cc[i]:
				continue
		
			score_best= 0
			index_best= -1
			n_merged= len(cc[i])

			for j in range(n_merged):
				index= cc[i][j]
				score= scores_det[index]
				if score>score_best:
					score_best= score
					index_best= index
				
			bboxes_sel.append(bboxes_det[index_best])
			labels_sel.append(labels_det[index_best])
			scores_sel.append(scores_det[index_best])
			class_ids_sel.append(class_ids_det[index_best])
		
			logger.info("Obj no. %d: bbox(%s), score=%f, class_id=%d, label=%s" % (i+1, str(bboxes_det[index_best]), scores_det[index_best], class_ids_det[index_best], labels_det[index_best]))
		
		logger.info("#%d selected objects left after merging overlapping ones ..." % len(bboxes_sel))
		self.bboxes_final= bboxes_sel
		self.scores_final= scores_sel
		self.labels_final= labels_sel
		self.class_ids_final= class_ids_sel
		
		return 0
		
	# ===========================
	# ==  DRAW
	# ===========================
	def draw_results(self, outfile):
		""" Draw results """
		
		# - Normalize image for drawing scopes to [0,255]
		img= self.image.copy()
		img_max= img.max()
		if img_max==1:
			img*= 255.
		img= img.astype(np.uint32)
		
		# - Draw
		figsize=(16,16)
		fig, ax = plt.subplots(1, figsize=figsize)
		#fig, ax = plt.subplots()
		
		# - Show area outside image boundaries
		title= self.image_path_base_noext
		height, width = self.image.shape[:2]
		
		ax.set_ylim(height + 2, -2)
		ax.set_xlim(-2, width + 2)
		ax.axis('off')
		
		ax.imshow(img)

		# - Draw bounding box rect
		for i in range(len(self.bboxes_final)):
			bbox= self.bboxes_final[i]
			score= self.scores_final[i]
			label= self.labels_final[i]
			color = self.class_color_map[label]
			
			# - Draw bounding box rect
			x1= bbox[0]
			y1= bbox[1]
			x2= bbox[2]
			y2= bbox[3]
			dx= x2-x1
			dy= y2-y1
			
			rect= patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, alpha=0.7, linestyle="solid", edgecolor=color, facecolor='none')
			ax.add_patch(rect)
			
			# Label
			if self.draw_class_label_in_caption:
				caption = "{} {:.2f}".format(label, score)
				ax.text(x1, y1 + 8, caption, color=color, size=20, backgroundcolor="none")
			else:
				caption = "{:.2f}".format(score)
				#ax.text(x1 + dx/2 - 4, y1 - 1, caption, color=color, size=23, backgroundcolor="none")
				#ax.text(x1 + dx/2 - 4, y1 - 1, caption, color="mediumturquoise", size=23, backgroundcolor="none")
				ax.text(x1 + dx/2 - 4, y1 - 1, caption, color="darkturquoise", size=30, backgroundcolor="none")

		# - Write to file	
		logger.info("Write plot to file %s ..." % outfile)
		if self.save_plots:
			fig.savefig(outfile)
			##fig.savefig(outfile, bbox_inches='tight')
			plt.close(fig)
		else:
			plt.show()

	

	# ====================================
	# ==   WRITE RESULTS IN JSON FORMAT
	# ====================================
	def make_json_results(self):
		""" Create a dictionary with detected objects """
		
		self.results= {
			"image_id": self.image_id, 
			"objs": []
		}

		xmin= self.image_xmin
		ymin= self.image_ymin
		imgshape= self.image.shape
		nx= imgshape[1]
		ny= imgshape[0]

		# - Loop over detected objects
		if self.bboxes_final:
			for i in range(len(self.bboxes_final)):
				# - Get detection info
				if self.obj_name_tag=="":
					sname= 'S' + str(i+1)
				else:
					sname= 'S' + str(i+1) + "_" + self.obj_name_tag
				class_id= self.class_ids_final[i]
				class_name= self.labels_final[i]
				###y1, x1, y2, x2 = self.bboxes_final[i]
				x1, y1, x2, y2 = self.bboxes_final[i]
				score= self.scores_final[i]

				x1= int(x1)
				x2= int(x2)
				y1= int(y1)
				y2= int(y2)
				class_id= int(class_id)

				at_edge= False
				if x1<=0 or x1>=nx-1 or x2<=0 or x2>=nx-1:
					at_edge= True
				if y1<=0 or y1>=ny-1 or y2<=0 or y2>=ny-1:
					at_edge= True

				d= {
					"name": str(sname),
					"x1": float(xmin + x1),
					"x2": float(xmin + x2),
					"y1": float(ymin + y1),
					"y2": float(ymin + y2),
					"class_id": int(class_id),
					"class_name": str(class_name),
					"score": float(score),
					"edge": int(at_edge)
				}
				self.results["objs"].append(d)
	
		
	def write_json_results(self, outfile):
		""" Write a json file with detected objects """
		
		# - Check if result dictionary is filled
		if not self.results:
			logger.warn("Result obj dictionary is empty, nothing to be written...")
			return
				
		# - Write to file
		with open(outfile, 'w') as fp:
			json.dump(self.results, fp, indent=2, sort_keys=True)
		
	# ====================================
	# ==   WRITE RESULTS IN DS9 FORMAT
	# ====================================
	def make_ds9_regions(self):
		""" Make a list of DS9 regions from json results """

		# - Check if result dictionary was created
		self.obj_regions= []
		if not self.results:
			logger.warn("No result dictionary was filled or no object detected, no region will be produced...")
			return -1
		if 'objs' not in self.results:
			logger.warn("No object list found in result dict...")
			return -1

		# - Loop over dictionary of detected object
		for detobj in self.results['objs']:
			sname= detobj['name']
			x1= detobj['x1']
			x2= detobj['x2']
			y1= detobj['y1']
			y2= detobj['y2']
			dx= x2-x1
			dy= y2-y1
			xc= x1 + 0.5*dx
			yc= y1 + 0.5*dy
			class_name= detobj['class_name']
			
			# - Set region tag
			at_edge= detobj['edge']
			##class_tag= '{' + class_name + '}'
			class_tag= class_name

			tags= []
			tags.append(class_tag)
			if at_edge:
				##tags.append('{BORDER}')
				tags.append('BORDER')

			color= self.class_color_map_ds9[class_name]
			
			rmeta= regions.RegionMeta({"text": sname, "tag": tags})
			rvisual= regions.RegionVisual({"color": color})
			r= regions.RectanglePixelRegion(PixCoord(xc, yc), dx, dy, meta=rmeta, visual= rvisual)
			self.obj_regions.append(r)
	

	def write_ds9_regions(self, outfile):
		""" Write DS9 region file """
	
		# - Check if region list is empty
		if not self.obj_regions:
			logger.warn("Region list with detected objects is empty, nothing to be written...")
			return

		# - Write to file
		try:
			regs_out= regions.Regions(self.obj_regions)
			regs_out.write(filename=outfile, format='ds9', overwrite=True) # available for version >=0.5
		except Exception as e:
			try:	
				logger.warn("Failed to write region list to file (err=%s), retrying with write_ds9 (<0.5 regions API) ..." % (str(e)))
				regions.write_ds9(regions=self.obj_regions, filename=outfile, coordsys='image') # this is to be used for versions <0.5 (deprecated in v0.5)
			except Exception as e:
				logger.warn("Failed to write region list to file (err=%s)!" % str(e))
			
	def write_fits(self, outfile):
		""" Write image in FITS file format """
		
		logger.info("Saving 2D image to file %s ..." % (outfile))
		utils.write_fits(self.image[:,:,0], outfile)
		
		
