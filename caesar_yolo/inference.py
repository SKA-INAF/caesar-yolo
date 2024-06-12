
# ===============================
#        MODULE IMPORTS
# ===============================
# - Parallel modules (this automatically calls MPI Init)
#from mpi4py import MPI

# - Standard modules
import os
import sys
import json
import time
import argparse
import datetime
import random
import numpy as np
from numpyencoder import NumpyEncoder

# - Image proc modules
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.stats import sigma_clipped_stats
import skimage.measure
from skimage.measure import find_contours
import cv2 as cv
from PIL import Image

# - Import regions module
import regions
from regions import PolygonPixelRegion, RectanglePixelRegion, PixCoord

# - Draw modules
import matplotlib.pyplot as plt

# - caesar-yolo modules
from caesar_yolo import logger
from caesar_yolo import utils
from caesar_yolo.evaluation import Analyzer
from caesar_yolo.graph import Graph

# ===============================
#     MergedSourceInfo CLASS
# ===============================
class MergedSourceInfo(object):
	""" Helper class for merging edge sources """
	def __init__(self, sindex, tindex):

		self.sindex= sindex
		self.tindex= tindex

# ===============================
#        TileTask CLASS
# ===============================
class TileTask(object):
	""" Define tile task object """

	def __init__(self, tile_coords, model, config):
		""" Return a tile task object """

		# - Config & model data
		self.model= model
		self.config= config

		# - Parallel options
		self.coords= tile_coords
		self.ix_min= tile_coords[0]
		self.ix_max= tile_coords[1]
		self.iy_min= tile_coords[2]
		self.iy_max= tile_coords[3]
		self.wid= -1
		self.tid= 0
		self.sname_tag= ""
		self.neighborTaskId= []
		self.neighborTaskIndex= []
		self.neighborWorkerId= []

		# - Image data
		self.imgdata= None
		self.imgheader= None
		img_fullpath= os.path.abspath(self.config['image_path'])
		img_path_base= os.path.basename(img_fullpath)
		img_path_base_noext= os.path.splitext(img_path_base)[0]
		self.image_id= img_path_base_noext

		# - Source data
		self.det_sources= {}
		self.bboxes_det= None
		self.scores_det= None	
		self.classid_det= None
		self.masks_det= None
	
		# - Save to file
		self.save_json= False
		self.save_regions= False
		self.save_img= False

	def set_worker_id(self, wid):
		""" Set worker ID for this tile task """
		self.wid= wid

	def set_task_id(self, tid):
		""" Set task ID for this tile task """
		self.tid= tid
		self.sname_tag= "t" + str(tid)

	def init_mpi(self):
		""" Init MPI parameters """
		
		if self.config['mpi'] is None:
			self.mpiEnabled= False
			self.nproc= 1
			self.mpi_version= ""
			self.procId= 0
		else:
			self.comm= self.config['mpi'].COMM_WORLD
			self.nproc= self.comm.Get_size()
			self.procId= self.comm.Get_rank()
			self.mpiEnabled= True

	def is_task_tile_adjacent(self, aTask):
		""" Check if this task tile is adjacent to another given task """

		ix_min_N= aTask.ix_min
		ix_max_N= aTask.ix_max
		iy_min_N= aTask.iy_min
		iy_max_N= aTask.iy_max

		isAdjacentInX= (self.ix_max==ix_min_N-1 or self.ix_min==ix_max_N+1 or (self.ix_min==ix_min_N and self.ix_max==ix_max_N))
		isAdjacentInY= (self.iy_max==iy_min_N-1 or self.iy_min==iy_max_N+1 or (self.iy_min==iy_min_N and self.iy_max==iy_max_N))
		isAdjacent= isAdjacentInX and isAdjacentInY
	
		return isAdjacent

	def is_task_tile_overlapping(self, aTask):
		""" Check if this task tile is overlapping with another given task """		
		
		ix_min_N= aTask.ix_min
		ix_max_N= aTask.ix_max
		iy_min_N= aTask.iy_min
		iy_max_N= aTask.iy_max

		if self.ix_max < ix_min_N: 
			return False # a is left of b
		if self.ix_min > ix_max_N: 
			return False # a is right of b
		if self.iy_max < iy_min_N: 
			return False # a is above b
		if self.iy_min > iy_max_N: 
			return False # a is below b
		
		return True
		
	
	def is_task_tile_neighbor(self, aTask):
		""" Check if given task is neighbour to this according to tile coordinates """		
	
		isOverlapping= self.is_task_tile_overlapping(aTask)
		isAdjacent= self.is_task_tile_adjacent(aTask)
		isNeighbor= (isAdjacent or isOverlapping)
		return isNeighbor

	def add_neighbor_info(self, tid, tindex, wid):
		""" Add neighbor info """

		self.neighborTaskId.append(tid)
		self.neighborTaskIndex.append(tindex)
		self.neighborWorkerId.append(wid)
		

	def find_sources(self):
		""" Read image and extract objects """

		# - Init MPI and other pars pars
		self.init_mpi()

		self.bboxes_det= None
		self.scores_det= None	
		self.classid_det= None

		# - Read image
		image_path= self.config['image_path']
		image_ext= os.path.splitext(image_path)[1]
		if image_ext!='.fits':
			logger.error("[PROC %d] Only FITS images are supported in parallel run (task %d), exit!" % (self.procId, self.tid))
			return -1
		
		res= utils.read_fits_crop(
			image_path, 
			ixmin= self.ix_min, ixmax= self.ix_max,
			iymin= self.iy_min, iymax= self.iy_max,
			strip_deg_axis=True
		)
		if res is None:
			logger.warn("[PROC %d] Failed to read tile image for task %d!" % (self.procId, self.tid))
			return -1

		self.imgdata, self.imgheader, wcs= res
		
		# - Set device used for inference. 
		#   By default it is set to setting used in config.
		#   If set_device_to_mpi_proc
		device= self.config['devices'][0]
		ndevices= len(self.config['devices'])
		if self.config['use_multi_gpu'] and ndevices>1:
			if self.nproc<=ndevices:# set device to same MPI process ID
				device= self.config['devices'][self.procId]
				logger.info("[PROC %d] Setting device for task %d to same proc id (device=%s) ..." % (self.procId, self.tid, device))
		
			else: # set device to a random GPU
				rand_index= random.randint(0, ndevices-1)
				device= self.config['devices'][rand_index]
				logger.info("[PROC %d] Setting device for task %d to a random device among given device list (index=%d, device=%s) ..." % (self.procId, self.tid, rand_index, device))
		
		logger.info("[PROC %d] Selected device %s for task %d..." % (self.procId, device, self.tid))

		# - Apply model
		analyzer= Analyzer(self.model, self.config)
		analyzer.device= device
		analyzer.draw= False
		analyzer.outfile= self.config['outfile']
		analyzer.write_to_json= self.save_json
		analyzer.outfile_json= 'catalog_' + self.image_id + '_tid' + str(self.tid) + '.json'
		analyzer.write_to_ds9= self.save_regions
		analyzer.outfile_ds9= 'catalog_' + self.image_id + '_tid' + str(self.tid) + '.reg'
		analyzer.save_img= self.save_img
		analyzer.outfile_img= 'timg_' + self.image_id + '_tid' + str(self.tid) + '.fits'
		analyzer.obj_name_tag= self.sname_tag

		if analyzer.predict(self.imgdata, self.image_id, header=self.imgheader, xmin=self.ix_min, ymin=self.iy_min)<0:
			logger.error("[PROC %d] Failed to run model prediction on tile image for task %d!" % (self.procId, self.tid))
			return -1

		# - Get raw and jsonized results
		#   NB: return if no object was detected
		bboxes_det= analyzer.bboxes_final
		if not bboxes_det:
			logger.info("[PROC %d] No object detected in tile image for task %d ..." % (self.procId, self.tid))
			return 0
		
		self.bboxes_det= bboxes_det
		self.scores_det= analyzer.scores_final	
		self.classid_det= analyzer.class_ids_final
		self.det_sources= analyzer.results

		# - Add tile info to detected source dictionary
		self.det_sources["workerId"]= self.wid
		self.det_sources["tileId"]= self.tid
		self.det_sources["neighborTileIds"]= self.neighborTaskId
		self.det_sources["xmin"]= self.ix_min
		self.det_sources["xmax"]= self.ix_max
		self.det_sources["ymin"]= self.iy_min
		self.det_sources["ymax"]= self.iy_max
	
		# - Print results
		logger.info("[PROC %d] #%d objects found in tile image for task %d ..." % (self.procId, len(bboxes_det), self.tid))
		#print("--> [PROC %d] bboxes_det (tid=%d)" % (self.procId, self.tid))
		#print(self.bboxes_det)
		#print("--> [PROC %d] scores_det (tid=%d)" % (self.procId, self.tid))
		#print(self.scores_det)
		#print("--> [PROC %d] classid_det (tid=%d)" % (self.procId, self.tid))
		#print(self.classid_det)
		#print("--> [PROC %d] masks_det (tid=%d)" % (self.procId, self.tid))
		#print(type(self.masks_det))
		#for mask in self.masks_det:
		#	print(type(mask))
		#	print(mask.shape)
		#	print(mask)
		#print("--> [PROC %d] det_sources (tid=%d)" % (self.procId, self.tid))
		#print(self.det_sources)


		return 0

# ===============================
#        SFinder CLASS
# ===============================
class SFinder(object):
	""" Define sfinder class """

	def __init__(self, model, config):
		""" Return a tile task object """

		# - Set model & config
		self.config= config
		self.model= model
		
		# - Init image/tile size parameters
		self.header= None
		self.wcs= None
		self.dX= 0
		self.dY= 0
		self.beamArea= 0
		self.pixelArea= 0
		self.bmaj= 0
		self.bmin= 0
		self.pa= 0
		self.image_id= ""
		self.nx= -1
		self.ny= -1
		self.read_subimg= False
		self.xmin= -1
		self.xmax= -1
		self.ymin= -1
		self.ymax= -1
		self.tileSizeX= -1
		self.tileSizeY= -1
		self.tileStepSizeX= 1
		self.tileStepSizeY= 1

		# - Init MPI vars
		self.mpiEnabled= False
		self.comm= None
		self.nproc= 1
		self.mpi_version= ""
		self.tasks_per_worker= []
		self.procId= 0
		self.MASTER_ID= 0
		self.mpiGroupsInitialized= True
		self.workerRanks = -1
		self.nWorkers = -1

		# - Source data
		self.tile_sources= {"sources": []}
		self.sources= {"sources": []}

		# - Save DS9 regions options
		self.save_tile_regions= config['save_tile_region']
		self.write_to_ds9= config['save_region']
		self.sregions= []
		self.outfile_ds9= ""
		self.class_color_map_ds9= {
			'bkg': "black",
			'spurious': "red",
			'compact': "blue",
			'extended': "green",
			'extended-multisland': "yellow",
			'flagged': "black",
			'diffuse': "magenta"
		}

		# - Save json catalog output file
		self.save_tile_json= config['save_tile_catalog']
		self.write_to_json= config['save_catalog']
		self.outfile_json= ""
		
		# - Save image FITS
		self.save_tile_img= config['save_tile_img']
		self.outfile_img= ""
		
		
	def set_img_size_params(self):
		""" Set image size parameters """
		
		# - Read image header
		image_path= self.config['image_path']
		image_ext= os.path.splitext(image_path)[1]
		
		self.header= None
		if image_ext=='.fits':
			self.header= utils.get_fits_header(image_path)
			if self.header is None:
				logger.error("[PROC %d] Header read from image %s is None!" % (self.procId, image_path))
				return -1

		# - Set image size
		xmin= self.config['image_xmin']
		xmax= self.config['image_xmax']
		ymin= self.config['image_ymin']
		ymax= self.config['image_ymax']

		if xmin>=0 and xmax>=0 and ymin>=0 and ymax>=0:
			self.read_subimg= True
			self.nx= (self.xmax-self.xmin+1)
			self.ny= (self.ymax-self.ymin+1)
			self.xmin= xmin
			self.xmax= xmax
			self.ymin= ymin
			self.ymax= ymax
		else:
			self.read_subimg= False

			if image_ext=='.fits':
				if 'NAXIS1' not in self.header:
					logger.error("[PROC %d] NAXIS1 keyword missing in header!" % (self.procId))
					return -1
				if 'NAXIS2' not in self.header:
					logger.error("[PROC %d] NAXIS2 keyword missing in header!" % (self.procId))
					return -1

				self.nx= self.header['NAXIS1']
				self.ny= self.header['NAXIS2']
			
			else:

				im= Image.open(image_path, mode='r')
				self.nx, self.ny = im.size


			self.xmin= 0
			self.xmax= self.nx-1
			self.ymin= 0
			self.ymax= self.ny-1

		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Image size (%d x %d), range=(%d,%d,%d,%d)" % (self.procId, self.nx, self.ny, self.xmin, self.xmax, self.ymin, self.ymax))
			
		# - Set tile parameters
		self.tileSizeX= self.nx
		self.tileSizeY= self.ny
		self.tileStepSizeX= 1
		self.tileStepSizeY= 1
		
		if self.config['split_image_in_tiles']:
			self.tileSizeX= self.config['tile_xsize']
			self.tileSizeY= self.config['tile_ysize']
			self.tileStepSizeX= self.config['tile_xstep']
			self.tileStepSizeY= self.config['tile_ystep']
			
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Tile size (%d x %d), step (%d,%d)" % (self.procId, self.tileSizeX, self.tileSizeY, self.tileStepSizeX, self.tileStepSizeY))
	
		img_fullpath= os.path.abspath(image_path)
		img_path_base= os.path.basename(img_fullpath)
		img_path_base_noext= os.path.splitext(img_path_base)[0]
		self.image_id= img_path_base_noext

		# - Compute beam area
		if image_ext=='.fits':
			compute_beam_area= True
			self.beamArea= 0

			if 'CDELT1' not in self.header:
				logger.warn("[PROC %d] CDELT1 keyword missing in header!" % (self.procId))
				compute_beam_area= False
			else:
				self.dX= self.header['CDELT1']

			if 'CDELT2' not in self.header:
				logger.error("[PROC %d] CDELT2 keyword missing in header!" % (self.procId))
				compute_beam_area= False	
			else:
				self.dY= self.header['CDELT2']

			if 'BMAJ' not in self.header:
				logger.warn("[PROC %d] BMAJ keyword missing in header!" % (self.procId))
				compute_beam_area= False
			else:
				self.bmaj= self.header['BMAJ']

			if 'BMIN' not in self.header:
				logger.warn("[PROC %d] BMIN keyword missing in header!" % (self.procId))
				compute_beam_area= False
			else:
				self.bmin= self.header['BMIN']
		
			if 'BPA' not in self.header:
				logger.warn("[PROC %d] BPA keyword missing in header!" % (self.procId))
				compute_beam_area= False
			else:
				self.pa= self.header['BPA']
		
			if compute_beam_area:
				self.pixelArea= np.abs(self.dX*self.dY)
				A= np.pi*self.bmaj*self.bmin/(4*np.log(2))
				self.beamArea= A/self.pixelArea
				if self.procId==self.MASTER_ID:
					logger.info("[PROC %d] Image info: beam(%f,%f,%f), beamArea=%f, dx=%f, dy=%f, pixArea=%g" % (self.procId, self.bmaj*3600, self.bmin*3600, self.pa, self.beamArea, self.dX*3600, self.dY*3600, self.pixelArea))

			# - Get WCS
			self.wcs = WCS(self.header)



		return 0
		

	

	####################################
	###      RUN
	####################################
	def run(self):
		""" Detect object in image """

		# - Set img size parameters
		if self.set_img_size_params()<0:
			logger.error("Failed to set image size parameters!")
			return -1

		# - Read image data
		image_path= self.config['image_path']
		image_ext= os.path.splitext(image_path)[1]
		logger.info("Reading image %s (ext=%s) ..." % (image_path, image_ext))
		
		if image_ext==".fits":
			res= utils.read_fits_crop(
				image_path, 
				self.config['image_xmin'], self.config['image_xmax'],
				self.config['image_ymin'], self.config['image_ymax'],
				strip_deg_axis=True
			)
			if res is None:
				logger.error("Failed to read image %s!" % (image_path))
				return -1
			
			image_data, header, wcs= res 
			
		elif image_ext==".png" or image_ext==".jpg":
			image_data= plt.imread(image_path)
			image_shape= image_data.shape
			ndim= image_data.ndim
			if ndim==3 and image_shape[2]==4:# remove alpha channel
				image_without_alpha = image_data[:,:,:3]
				image_data= image_without_alpha
				
			header= None
						
		else:
			logger.error("Unsupported image format (%s) given!" % (image_ext))
			return -1
		

		img_fullpath= os.path.abspath(image_path)
		img_path_base= os.path.basename(img_fullpath)
		img_path_base_noext= os.path.splitext(img_path_base)[0]
		self.image_id= img_path_base_noext
	
		# - Apply model 
		analyzer= Analyzer(self.model, self.config)
		
		if analyzer.predict(image=image_data, image_id=self.image_id, header=header)<0:
			logger.error("Failed to run model prediction on image %s!" % (image_path))
			return -1
		
		# - Get results
		bboxes_det= analyzer.bboxes_final
		scores_det= analyzer.scores_final	
		classid_det= analyzer.class_ids_final
		
		# - Return if no object was detected
		if not bboxes_det:
			logger.info("No object detected in image %s ..." % (image_path))
			return 0
	
		# - Print results
		logger.info("#%d objects found in image %s ..." % (len(bboxes_det), image_path))
		

		return 0

	####################################
	###      RUN PARALLEL
	####################################
	def init_mpi(self):
		""" Init MPI parameters """
		
		# - Init MPI (module import automatically calls MPI Init)		
		if self.config['mpi'] is None:
			logger.warn("MPI instance is None, running in serial ...")
			self.mpiEnabled= False
			self.nproc= 1
			self.mpi_version= ""
			self.procId= 0
		else:
			self.comm= self.config['mpi'].COMM_WORLD
			self.nproc= self.comm.Get_size()
			self.mpi_version= self.config['mpi'].Get_version()
			self.procId= self.comm.Get_rank()
			self.mpiEnabled= True

		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Using #%d processors for this run (MPI enabled? %d, version=%s) ..." % (self.procId, self.nproc, self.mpiEnabled, self.mpi_version))

		
	def run_parallel(self):
		""" Detect object in image (dividing computation across multiple tiles) """

		# - Init MPI
		self.init_mpi()

		if self.comm:
			self.comm.Barrier()
		t0= time.time()
		
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] MPI enabled? %d " % (self.procId, self.mpiEnabled))

		# - Set image size parameters
		#   NB: Executed by all PROC
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Setting image size parameters ..." % (self.procId))
		if self.set_img_size_params()<0:
			logger.error("[PROC %d] Failed to set image size parameters!" % (self.procId))
			return -1

		# - Partition tile tasks among workers
		#   NB: Executed by all PROC
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Partitioning tile tasks among workers ..." % (self.procId))
		if self.create_tile_tasks()<0:
			logger.warn("[PROC %d] Failure in create tile tasks, exit..." % (self.procId))
			return -1

		#  - Extract sources in tiles assigned to current worker
		status= 0
		nTasks= len(self.tasks_per_worker[self.procId])

		for j in range(nTasks):
			tid= self.tasks_per_worker[self.procId][j].tid
			logger.info("[PROC %d] Start processing of task id %d (%d/%d) ..." % (self.procId, tid, j+1, nTasks))
		
			if self.tasks_per_worker[self.procId][j].find_sources()<0:
				logger.error("[PROC %d] Failed to find sources in task no. %d, skip to next!" % (self.procId, j))
				status= -1
				continue

			# - Find sources at edges in this tile
			logger.info("[PROC %d] Findind sources at edge in task id %d (%d/%d) ..." % (self.procId, tid, j+1, nTasks))
			self.find_sources_at_edge(j)
	
		if status<0:
			logger.warn("[PROC %d] One or more errors occurred in source finding tasks..." % (self.procId))

		# - Gather all sources from all tiles
		#		Update task info (tile physical range) from workers
		# 	The updated list of task data is available in master processor
		#   (check if it is better to replace with MPI_Gather and put it available in all workers)
		if self.mpiEnabled:
			if self.procId==self.MASTER_ID:
				logger.info("[PROC %d] Gathering task data from all workers ..." % (self.procId))

			if self.gather_task_data_from_workers()<0:
				logger.error("[PROC %d] Gathering task data from workers failed!" % (self.procId))
				return -1
			
		# - Merge edge sources
		#   NB: Done only by master proc
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Merging sources at tile edges ..." % self.procId)
			self.merge_edge_sources()

		# - Save to file
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Saving source results to file ..." % self.procId)
			self.save()

		# - Stop timer and count runtime
		if self.mpiEnabled:
			self.comm.Barrier()
		t1= time.time()
		runtime= t1-t0
		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] Run completed in %d seconds" % (self.procId, runtime))

		return 0

	####################################
	###      FIND SOURCES AT EDGE
	####################################
	def find_sources_at_edge(self, tindex):
		""" Find sources at tile edges or overlap region """
		
		# - Check if there are sources found in this tile
		tileData= self.tasks_per_worker[self.procId][tindex]
		sourceTileData= tileData.det_sources
		if not sourceTileData:
			logger.warn("[PROC %d] No source data for this tile, nothing to be done..." % self.procId)
			return

		sources= sourceTileData["objs"]
		if not sources:
			logger.warn("[PROC %d] No sources for this tile, nothing to be done..." % self.procId)
			return

		tid= tileData.tid
		xmin= tileData.ix_min
		xmax= tileData.ix_max
		ymin= tileData.iy_min
		ymax= tileData.iy_max
		neighborTaskIndexes= tileData.neighborTaskIndex
		neighborWorkerIndexes= tileData.neighborWorkerId

		# - Loop over sources and check if they are at edge or overlapping
		for i in range(len(sources)):
			source= sources[i]
			sname= source["name"]
			xmin_s= source["x1"]
			xmax_s= source["x2"]
			ymin_s= source["y1"]
			ymax_s= source["y2"]

			# - Check if at boundary
			isAtBoxEdgeX= (xmin_s==xmin or xmax_s==xmax)
			isAtBoxEdgeY= (ymin_s==ymin or ymax_s==ymax)
			isAtBoxEdge= (isAtBoxEdgeX or isAtBoxEdgeY)
			if isAtBoxEdge:
				logger.info("[PROC %d] Setting edge flag for source %s in tile %d as at border ..." % (self.procId, sname, tid))
				self.tasks_per_worker[self.procId][tindex].det_sources["objs"][i]["edge"]= True
				continue

			# - Check if in tile overlapping area
			for j in range(len(neighborWorkerIndexes)):
				tindex_n= neighborTaskIndexes[j]
				windex_n= neighborWorkerIndexes[j]
					
				tid_n= self.tasks_per_worker[windex_n][tindex_n].tid
				xmin_n= self.tasks_per_worker[windex_n][tindex_n].ix_min
				xmax_n= self.tasks_per_worker[windex_n][tindex_n].ix_max
				ymin_n= self.tasks_per_worker[windex_n][tindex_n].iy_min
				ymax_n= self.tasks_per_worker[windex_n][tindex_n].iy_max

				not_overlapping= ( 
					(xmax_s < xmin_n) or
					(xmin_s > xmax_n) or
					(ymax_s < ymin_n) or
					(ymin_s > ymax_n)
				)
				if not_overlapping:
					continue

				logger.info("[PROC %d] Setting edge flag for source %s in tile %d as found in overlapping region with tile %d ..." % (self.procId, sname, tid, tid_n))
				self.tasks_per_worker[self.procId][tindex].det_sources["objs"][i]["edge"]= True
				break
	
	####################################
	###      MERGE EDGE SOURCES
	####################################
	def merge_edge_sources(self):
		""" Merge sources at tile edges """

		# - Return if proc is not the master
		if self.procId!=self.MASTER_ID:
			return 0

		# - Fill list of edge sources and final merged sources (not at edge)
		sourcesToBeMerged= []
		self.sources["sources"]= []
		for tile_index in range(len(self.tile_sources["sources"])):
			tileData= self.tile_sources["sources"][tile_index]
			sources= tileData["objs"]
			for j in range(len(sources)):
				at_edge= sources[j]["edge"]
				if not at_edge:
					sources[j]["merged"]= False
					self.sources["sources"].append(sources[j])
					continue
				sourcesToBeMerged.append( MergedSourceInfo(j,tile_index) )

		# - Create graph with mergeable sources at edge
		N= len(sourcesToBeMerged)
		g= Graph(N)
		logger.info("[PROC %d] #%d sources at edge to be searched for merging ..." % (self.procId, N))

		for i in range(N):
			sindex= sourcesToBeMerged[i].sindex
			tindex= sourcesToBeMerged[i].tindex
			source= self.tile_sources["sources"][tindex]["objs"][sindex]
			classLabel= source["class_name"]
			wid= self.tile_sources["sources"][tindex]["workerId"]
			tid= self.tile_sources["sources"][tindex]["tileId"]
			tids_neighbor= self.tile_sources["sources"][tindex]["neighborTileIds"]
			sname= source["name"]
			xmin= source["x1"] 
			xmax= source["x2"]
			ymin= source["y1"] 
			ymax= source["y2"]
			##pixels= source["pixels"]

			logger.info("[PROC %d] Searching for sources adjacent/overlapping to source %s (wid=%d, tid=%d) ..." % (self.procId, sname, wid, tid))
			print("--> neighbors")
			print(tids_neighbor)
	
			for j in range(i+1,N):
				sindex_j= sourcesToBeMerged[j].sindex
				tindex_j= sourcesToBeMerged[j].tindex
				source_j= self.tile_sources["sources"][tindex_j]["objs"][sindex_j]
				classLabel_j= source_j["class_name"]
				wid_j= self.tile_sources["sources"][tindex_j]["workerId"]
				tid_j= self.tile_sources["sources"][tindex_j]["tileId"]
				sname_j= source_j["name"]
				xmin_j= source_j["x1"] 
				xmax_j= source_j["x2"]
				ymin_j= source_j["y1"] 
				ymax_j= source_j["y2"]
				##pixels_j= source_j["pixels"]
				
				# - Skip if tid_j is not among neighbor tiles
				if tid_j not in tids_neighbor:
					logger.info("[PROC %d] Skipping overlap check for source %s (wid=%d, tid=%d) as located in a non-neighbor tile ..." % (self.procId, sname_j, wid_j, tid_j))
					continue
	
				# - Skip merging if bounding boxes are not overlapping
				bbox_not_overlapping= (
					(xmax < xmin_j) or (xmin > xmax_j) or (ymax < ymin_j) or (ymin > ymax_j)
				)
				if bbox_not_overlapping:
					logger.info("[PROC %d] Skipping source %s (wid=%d, tid=%d) as bbox [%d,%d,%d,%d] is not overlapping with [%d,%d,%d,%d] ..." % (self.procId, sname_j, wid_j, tid_j, xmin_j, xmax_j, ymin_j, ymax_j, xmin, xmax, ymin, ymax))
					continue

				logger.info("[PROC %d] Sources %s (wid=%d, tid=%d) and %s (wid=%d, tid=%d) have overlapping bounding boxes, selected for merging  ..." % (self.procId, sname, wid, tid, sname_j, wid_j, tid_j))

				g.addEdge(i,j)
				
		# - Find all connected sources in graph and merge them
		cc = g.connectedComponents()
		for i in range(len(cc)):
			if not cc[i]:
				continue

			sname_merged= "S" + str(i+1) + "_merged" 
			n_merged= len(cc[i])	

			# - If >=2 sources are to be merged, assign class and score of the biggest one (Npix)
			#   If only one source is left, add to the list of final sources
			if n_merged==1:
				index= cc[i][0]
				sindex= sourcesToBeMerged[index].sindex
				tindex= sourcesToBeMerged[index].tindex
				wid= self.tile_sources["sources"][tindex]["workerId"]
				tid= self.tile_sources["sources"][tindex]["tileId"]
				source= self.tile_sources["sources"][tindex]["objs"][sindex]
				sname= source["name"]
				source["name"]= sname_merged
				source["merged"]= False
				self.sources["sources"].append(source)
				logger.info("[PROC %d] Adding single edge source %s (wid=%d, tid=%d) to list with name %s ..." % (self.procId, sname, wid, tid, sname_merged))

			else: 
				index_largest= -1
				bbox_area_largest= -1
				bboxes_to_be_merged= []
				###npix_largest= -1
				###pixels_merged= []
				
				for j in range(n_merged):
					index= cc[i][j]
					sindex= sourcesToBeMerged[index].sindex
					tindex= sourcesToBeMerged[index].tindex
					source= self.tile_sources["sources"][tindex]["objs"][sindex]
					x1= source["x1"] 
					x2= source["x2"]
					y1= source["y1"] 
					y2= source["y2"]
					bbox= (x1,y1,x2,y2)
					bbox_area= (x2-x1)*(y2-y1)
					if bbox_area>bbox_area_largest:
						bbox_area_largest= bbox_area
						index_largest= index
				
					bboxes_to_be_merged.append(bbox)
				
					###pixels= source["pixels"]
					###npix= len(pixels)
					###if npix>npix_largest:
					###	npix_largest= npix
					###	index_largest= index

					# - Merge pixels (without duplicates)
					###merged= pixels_merged + [x for x in pixels if x not in pixels_merged]
					###pixels_merged= merged

				# - Set class & score of merged source
				sindex_largest= sourcesToBeMerged[index_largest].sindex
				tindex_largest= sourcesToBeMerged[index_largest].tindex
				source_largest= self.tile_sources["sources"][tindex_largest]["objs"][sindex_largest]
				score_merged= source_largest["score"]
				className_merged= source_largest["class_name"]
				classId_merged= source_largest["class_id"]

				# - Compute new bbox (merge of individual bboxes)
				bbox_merged= utils.get_merged_bbox(bboxes_to_be_merged)
				x1_merged= bbox_merged[0]
				y1_merged= bbox_merged[1]
				x2_merged= bbox_merged[2]
				y2_merged= bbox_merged[3]
				
				#pix_min= np.min(pixels_merged,axis=0)
				#pix_max= np.max(pixels_merged,axis=0)
				#xmin= pix_min[1]
				#xmax= pix_max[1]
				#ymin= pix_min[0]
				#ymax= pix_max[0]
				#dx= xmax-xmin+1
				#dy= ymax-ymin+1

				# - Compute new vertices
				#offset= 10
				#padded_mask= np.zeros((dy+2*offset,dx+2*offset),dtype=np.uint8)
				#pp= np.array(pixels_merged) - [ymin,xmin]
				#for item in pp:
				#	padded_mask[item[0]+offset,item[1]+offset]= 1				

				#contours = find_contours(padded_mask, 0.5)
				#vertexes= []
				#for verts in contours:
				#	# - Subtract the padding and flip (y, x) to (x, y)
				#	verts = np.fliplr(verts)
				#	vertexes.append(verts.tolist())

				#vertex_list= vertexes
				#for k in range(len(vertex_list)):
				#	for nvert in range(len(vertex_list[k])):
				#		vertex_list[k][nvert][0]+= xmin-offset
				#		vertex_list[k][nvert][1]+= ymin-offset

				# - Make merged source and append to list
				source_merged= {}
				source_merged["name"]= str(sname_merged)
				source_merged["x1"]= float(x1_merged)
				source_merged["x2"]= float(x2_merged)
				source_merged["y1"]= float(y1_merged)
				source_merged["y2"]= float(y2_merged)
				source_merged["edge"]= True
				source_merged["merged"]= True
				source_merged["score"]= score_merged
				source_merged["class_name"]= className_merged
				source_merged["class_id"]= classId_merged
				
				self.sources["sources"].append(source_merged)

		# - Create final list of extracted sources
		nsources= len(self.sources["sources"])
		logger.info("[PROC %s] Renaming #%d extracted sources ..." % (self.procId, nsources))
		for i in range(nsources):
			sname= "S" + str(i+1)
			self.sources["sources"][i]["name"]= sname

		return 0

	####################################
	###      GATHER SOURCE TILE DATA
	####################################
	def gather_task_data_from_workers(self):
		""" Gather task data from all worker in master proc """

		# - Merge source data for all tasks in this worker
		#   NB: Executed by all workers
		logger.info("[PROC %d] Merging source data extracted in each tile by this worker ..." % (self.procId))
		self.tile_sources= {"sources": []}
		nTasks= len(self.tasks_per_worker[self.procId])
		for j in range(nTasks):
			sourceTileData= self.tasks_per_worker[self.procId][j].det_sources
			if sourceTileData:
				self.tile_sources["sources"].append(sourceTileData)

		# - Synchronize all workers
		logger.info("[PROC %d] Synchronize worker before data aggregation ..." % self.procId)
		self.comm.Barrier()
		
		# - Merge all sources found by workers in a unique collection
		MSG_TAG= 1
		if self.procId == self.MASTER_ID: # Receive data from the workers

			logger.info("[PROC %d] Gathering tile task data found by all workers ..." % (self.procId))
		
			for i in range(1,self.nproc):
				# Check if this processor has tasks assigned, otherwise skip!
				if not self.tasks_per_worker[i]:
					logger.info("[PROC %d] No tasks assigned to process %d, nothing to be collected, skip to next worker..." % (self.procId, i))
					continue
  
				# Receive data from master proc and append to master collection
				logger.info("[PROC %d] Receiving data from proc %d ..." % (self.procId, i))
				recv_data= self.comm.recv(source=i, tag=MSG_TAG)
				self.tile_sources["sources"].extend(recv_data["sources"])
				logger.info("[PROC %d] Data received from proc %d ..." % (self.procId, i))
				
			#logger.info("[PROC %d] Tile aggregated sources " % self.procId)
			#print(self.tile_sources)
		
		else: # Send aggregated data to master process

			logger.info("[PROC %d] Sending tile task data found to master process ..." % (self.procId))
			self.comm.send(self.tile_sources, dest=self.MASTER_ID, tag=MSG_TAG)
			logger.info("[PROC %d] Data sent to master process ..." % self.procId)
			
		# - Synchronize workers again
		logger.info("[PROC %d] Synchronize worker after data aggregation ..." % self.procId)
		self.comm.Barrier()

		return 0


	

	####################################
	###      CREATE TILE TASKS
	####################################
	def create_tile_tasks(self):
		""" Create tile tasks """
		
		# - Generate image partition in tiles given tile parameters
		tileGrid= utils.generate_tiles(
			self.xmin, self.xmax, 
			self.ymin, self.ymax, 
			self.tileSizeX, self.tileSizeY, 
			self.tileStepSizeX, self.tileStepSizeY
		)

		if self.procId==self.MASTER_ID:
			logger.info("[PROC %d] #%d tile tasks to be distributed among worker ..." % (self.procId, len(tileGrid)))
			print(tileGrid)

		# - Assign tile tasks to workers
		self.tasks_per_worker= tasks_per_worker= [[] for i in range(self.nproc)]
		workerCounter= 0

		for i in range(len(tileGrid)):
			coords= tileGrid[i]
			tid= i
			if self.procId==self.MASTER_ID:
				logger.info("[PROC %d] Assign tile task %d to worker no. %d ..." % (self.procId, tid, workerCounter))

			tileTask= TileTask(coords, self.model, self.config)
			tileTask.set_worker_id(workerCounter)
			tileTask.set_task_id(tid)
			tileTask.save_regions= self.save_tile_regions
			tileTask.save_json= self.save_tile_json
			tileTask.save_img= self.save_tile_img
			
			self.tasks_per_worker[workerCounter].append(tileTask)

			if workerCounter>=self.nproc-1: 
				workerCounter= 0
			else:
				workerCounter+= 1

		# - Fill neighbour tile task list
		workerIds= []

		for i in range(len(self.tasks_per_worker)): # loop over workers
			if not self.tasks_per_worker[i]:
				continue

			# Add only processors with tasks
			workerIds.append(i)
	
			# Loop over tasks present in this worker
			nTasksInWorker= len(self.tasks_per_worker[i])
		
			for j in range(nTasksInWorker): # loop over tasks
				task= self.tasks_per_worker[i][j]
				tid= task.tid

				# Find first neighbors among tasks inside the same worker
				for k in range(j+1, nTasksInWorker):
					if j==k: 
						continue
					task_N= self.tasks_per_worker[i][k]
					tid_N= task_N.tid
					areNeighbors= task.is_task_tile_neighbor(task_N)
					if areNeighbors:
						#self.tasks_per_worker[i][j].add_neighbor_info(k,i)
						#self.tasks_per_worker[i][k].add_neighbor_info(j,i)
						self.tasks_per_worker[i][j].add_neighbor_info(tid_N,k,i)
						self.tasks_per_worker[i][k].add_neighbor_info(tid,j,i)
				
				# Find neighbors across workers
				for s in range(i+1,len(self.tasks_per_worker)): 
					for t in range(len(self.tasks_per_worker[s])):
						task_N= self.tasks_per_worker[s][t]
						tid_N= task_N.tid
						areNeighbors= task.is_task_tile_neighbor(task_N)
						if areNeighbors:
							#self.tasks_per_worker[i][j].add_neighbor_info(t,s)
							#self.tasks_per_worker[s][t].add_neighbor_info(j,i)
							self.tasks_per_worker[i][j].add_neighbor_info(tid_N,t,s)
							self.tasks_per_worker[s][t].add_neighbor_info(tid,j,i)

		nWorkers= len(workerIds)	
		ss= "# " + str(nWorkers) + " workers {"
		for i in range(nWorkers):
			ss+= str(workerIds[i]) + ","
		ss+= "}"
		if self.procId==self.MASTER_ID:
			logger.info(ss)		

		
		# - Create a worker group (if MPI run is performed)
		self.workerRanks= -1
		self.nWorkers= 1

		if self.mpiEnabled:
			# Get main processor group
			self.worldGroup= self.comm.Get_group()
			
			# Construct a group containing all of the workers (proc with tasks assigned)
			self.workerGroup= self.config['mpi'].Group.Incl(self.worldGroup, workerIds)
			if self.procId==self.MASTER_ID:
				logger.info("[PROC %d] Worker group size: %d" % (self.procId, self.workerGroup.size))

			# Create a new communicator based on the group
			commTag= 10
			self.workerComm= self.comm.Create_group(self.workerGroup, commTag)

			self.mpiGroupsInitialized= True
			self.workerRanks = -1
			self.nWorkers = -1
	
			# If this rank isn't in the new communicator, it will be
			#	MPI_COMM_NULL. Using MPI_COMM_NULL for MPI_Comm_rank or
			# MPI_Comm_size is erroneous
			#if self.workerComm!=MPI_COMM_NULL:
			if self.workerComm is not None:
				self.workerRanks= self.workerComm.Get_rank()
				self.nWorkers= self.workerComm.Get_size()
			else:
				logger.warn("[PROC %d] Worker MPI communicator is null (this processor has no tasks and was not inserted in the worker group)!" % (self.procId))

		if self.procId==self.MASTER_ID:
			logger.info("WORLD RANK/SIZE: %d/%d, WORKER RANK/SIZE: %d/%d" % (self.procId, self.nproc, self.workerRanks, self.nWorkers))


		# - Print workers
		if self.procId==self.MASTER_ID:
			for i in range(len(self.tasks_per_worker)):
		
				if not self.tasks_per_worker[i]:
					continue # no tasks present

				for j in range(len(self.tasks_per_worker[i])):
					ss= "Worker no. " + str(i) + ", "
				
					ix_min= self.tasks_per_worker[i][j].ix_min
					ix_max= self.tasks_per_worker[i][j].ix_max
					iy_min= self.tasks_per_worker[i][j].iy_min
					iy_max= self.tasks_per_worker[i][j].iy_max

					ss+= "Task no. " + str(j) + " [" + str(ix_min) + "," + str(ix_max) + "," + str(iy_min) + "," + str(iy_max) + "] --> neighbors{"
			
					for k in range(len(self.tasks_per_worker[i][j].neighborTaskIndex)):

						neighborWorkerId= self.tasks_per_worker[i][j].neighborWorkerId[k]
						neighborTaskIndex= self.tasks_per_worker[i][j].neighborTaskIndex[k]
						neighborTaskId= self.tasks_per_worker[i][j].neighborTaskId[k]
						next_ix_min= self.tasks_per_worker[neighborWorkerId][neighborTaskIndex].ix_min
						next_ix_max= self.tasks_per_worker[neighborWorkerId][neighborTaskIndex].ix_max
						next_iy_min= self.tasks_per_worker[neighborWorkerId][neighborTaskIndex].iy_min
						next_iy_max= self.tasks_per_worker[neighborWorkerId][neighborTaskIndex].iy_max

						ss+= "(" + str(neighborWorkerId) + ","+ str(neighborTaskId) + ") [" + str(next_ix_min) + "," + str(next_ix_max) + "," + str(next_iy_min) + "," + str(next_iy_max) + "], "
				
					ss+= "}"
					if self.procId==self.MASTER_ID:
						logger.info(ss)	
		
		# - Check if max tasks per tile is exceeded
		hasTooManyTasks= False
		for i in range(len(self.tasks_per_worker)):
			nTasksPerWorker= len(self.tasks_per_worker[i])
			if nTasksPerWorker>self.config['max_ntasks_per_worker']:
				hasTooManyTasks= True
				break

		if hasTooManyTasks:
			logger.warn("[PROC %d] Too many tasks per worker exceeded (thr=%d)!" % (self.procId, self.config['max_ntasks_per_worker']))
			return -1
	
		return 0

	####################################
	###      SAVE
	####################################
	def save(self):
		""" Save extracted source to file """
	
		# - Return if called by other processor than MASTER
		if self.procId!=self.MASTER_ID:
			return

		# - Write json results?
		if self.write_to_json:
			logger.info("[PROC %d] Writing results for image %s to json ..." % (self.procId, str(self.image_id)))
			if self.outfile_json=="":
				outfile_json= 'catalog_' + str(self.image_id) + '.json'
			else:
				outfile_json= self.outfile_json
			self.write_json_results(outfile_json)

		# - Create DS9 region objects
		logger.info("[PROC %d] Creating DS9 regions from extracted sources ..." % self.procId)
		self.make_ds9_regions()

		# - Write DS9 regions to file?
		if self.write_to_ds9:
			logger.info("[PROC %d] Writing extracted source DS9 regions to file ..." % self.procId)
			if self.outfile_ds9=="":
				outfile_ds9= 'ds9_' + str(self.image_id) + '.reg'
			else:
				outfile_ds9= self.outfile_ds9
			self.write_ds9_regions(outfile_ds9)
	

	def write_json_results(self, outfile):
		""" Write a json file with detected objects """
		
		# - Return if called by other processor than MASTER
		if self.procId!=self.MASTER_ID:
			return

		# - Check if result dictionary is filled
		if not self.sources:
			logger.warn("[PROC %d] Source dictionary is empty, nothing to be written ..." % self.procId)
			return
				
		# - Write to file
		with open(outfile, 'w') as fp:
			json.dump(self.sources, fp, indent=2, sort_keys=True, cls=NumpyEncoder)


	def make_ds9_regions(self):
		""" Make a list of DS9 regions from json results """

		# - Return if called by other processor than MASTER
		if self.procId!=self.MASTER_ID:
			return

		# - Check if result dictionary is filled
		if not self.sources:
			logger.warn("[PROC %d] Source dictionary is empty, nothing to be produced ..." % self.procId)
			return

		# - Loop over dictionary of detected object
		self.sregions= []

		for detobj in self.sources['sources']:
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

			# - Set region metadata
			at_edge= detobj['edge']
			merged= detobj['merged']
			##class_tag= '{' + class_name + '}'
			class_tag= class_name

			tags= []
			tags.append(class_tag)
			if at_edge:
				tags.append('BORDER')
				##tags.append('{BORDER}')
				
			if merged:
				tags.append('MERGED')
				##tags.append('{MERGED}')

			color= self.class_color_map_ds9[class_name]
			
			rmeta= regions.RegionMeta({"text": sname, "tag": tags})
			rvisual= regions.RegionVisual({"color": color})
			r= regions.RectanglePixelRegion(PixCoord(xc, yc), dx, dy, meta=rmeta, visual= rvisual)

			self.sregions.append(r)


	def write_ds9_regions(self, outfile):
		""" Write DS9 region file """
	
		# - Return if called by other processor than MASTER
		if self.procId!=self.MASTER_ID:
			return

		# - Check if region list is empty
		if not self.sregions:
			logger.warn("[PROC %d] Region list with detected objects is empty, nothing to be written ..." % self.procId)
			return

		# - Write to file
		try:
			regs_out= regions.Regions(self.sregions)
			regs_out.write(filename=outfile, format='ds9', overwrite=True) # available for version >=0.5
		except:
			try:	
				logger.debug("[PROC %d] Failed to write region list to file, retrying with write_ds9 (<0.5 regions API) ..." % self.procId)
				regions.write_ds9(regions=self.sregions, filename=outfile, coordsys='image') # this is to be used for versions <0.5 (deprecated in v0.5)
			except Exception as e:
				logger.warn("[PROC %d] Failed to write region list to file (err=%s)!" % (self.procId, str(e)))
			


