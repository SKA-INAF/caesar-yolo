#!/usr/bin/env python

from __future__ import print_function

##################################################
###          MODULE IMPORT
##################################################
## STANDARD MODULES
import os
import sys
import subprocess
import string
import time
import signal
from threading import Thread
import datetime
import numpy as np
import random
import math
import logging
from collections import Counter
import json

## ASTROPY MODULES 
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
from astropy.stats import sigma_clip
from astropy.visualization import ZScaleInterval, MinMaxInterval, PercentileInterval, HistEqStretch

## SKIMAGE
from skimage.util import img_as_float64
from skimage.exposure import adjust_sigmoid, rescale_intensity, equalize_hist, equalize_adapthist


## PACKAGE MODULES
from .utils import compose_fcns, resize_img_v2

##############################
##     GLOBAL VARS
##############################
from caesar_yolo import logger


##############################
##     PREPROCESSOR CLASS
##############################
class DataPreprocessor(object):
	""" Data pre-processor class """

	def __init__(self, stages):
		""" Create a data pre-processor object """
	
		# - stages is a list of pre-processing instances (e.g. MinMaxNormalizer, etc).
		#   NB: First element is the first stage to be applied to data.
		self.fcns= [] # list of pre-processing functions
		for stage in stages: 
			self.fcns.append(stage.__call__)

		# - Reverse list as fcn compose take functions in the opposite order
		self.fcns.reverse()

		# - Create pipeline
		self.pipeline= compose_fcns(*self.fcns)

	def __call__(self, data):
		""" Apply sequence of pre-processing steps """
		return self.pipeline(data)

	

	
##############################
##     MinMaxNormalizer
##############################
class MinMaxNormalizer(object):
	""" Normalize each image channel to range  """

	def __init__(self, norm_min=0, norm_max=1, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.norm_min= norm_min
		self.norm_max= norm_max


	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Normalize data
		data_norm= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond]
			if data_ch_1d.size==0:
				logger.warn("Size of data_ch%d is zero, returning None!" % (i))
				return None

			data_ch_min= data_ch_1d.min()
			data_ch_max= data_ch_1d.max()
			data_ch_norm= (data_ch-data_ch_min)/(data_ch_max-data_ch_min) * (self.norm_max-self.norm_min) + self.norm_min
			data_ch_norm[~cond]= 0 # Restore 0 and nans set in original data
			data_norm[:,:,i]= data_ch_norm

		return data_norm

##############################
##   AbsMinMaxNormalizer
##############################
class AbsMinMaxNormalizer(object):
	""" Normalize each image channel to range using absolute min/max among all channels and not per-channel """

	def __init__(self, norm_min=0, norm_max=1, **kwparams):
		""" Create a data pre-processor object """
			
		# - Set parameters
		self.norm_min= norm_min
		self.norm_max= norm_max

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find absolute min & max across all channels
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_masked= np.ma.masked_where(~cond, data, copy=False)
		data_min= data_masked.min()
		data_max= data_masked.max()

		# - Normalize data
		data_norm= (data-data_min)/(data_max-data_min) * (self.norm_max-self.norm_min) + self.norm_min
		data_norm[~cond]= 0 # Restore 0 and nans set in original data
		
		return data_norm



##############################
##   MaxScaler
##############################
class MaxScaler(object):
	""" Divide each image channel by their maximum value """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find max for each channel
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_masked= np.ma.masked_where(~cond, data, copy=False)
		data_max= data_masked.max(axis=(0,1)).data

		# - Scale data
		data_scaled= data/data_max
		data_scaled[~cond]= 0 # Restore 0 and nans set in original data
		
		return data_scaled


##############################
##   AbsMaxScaler
##############################
class AbsMaxScaler(object):
	""" Divide each image channel by their absolute maximum value """

	def __init__(self, use_mask_box=False, mask_fract=0.5, **kwparams):
		""" Create a data pre-processor object """

		self.use_mask_box= use_mask_box
		self.mask_fract= mask_fract

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find absolute max
		#   NB: Excluding masked pixels (=0, & NANs)
		cond= np.logical_and(data!=0, np.isfinite(data))

		if self.use_mask_box:
			data_shape= data.shape
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			border_mask= np.zeros(data.shape)
			border_mask[ymin:ymax, xmin:xmax, :]= 1
			cond_max= np.logical_and(cond, border_mask==1)
		else:
			cond_max= cond

		data_masked= np.ma.masked_where(~cond_max, data, copy=False)
		data_max= data_masked.max()

		# - Scale data
		data_scaled= data/data_max
		data_scaled[~cond]= 0 # Restore 0 and nans set in original data
		
		return data_scaled


##############################
##   ChanMaxScaler
##############################
class ChanMaxScaler(object):
	""" Divide each image channel by selected channel maximum value """

	def __init__(self, chref=0, use_mask_box=False, mask_fract=0.5, **kwparams):
		""" Create a data pre-processor object """

		self.chref= chref
		self.use_mask_box= use_mask_box
		self.mask_fract= mask_fract

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Find selected channel max
		#   NB: Excluding masked pixels (=0, & NANs)
		data_ch= data[:,:,self.chref]
		if self.use_mask_box:
			data_shape= data_ch.shape
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.debug("Using box x[%d,%d] y[%d,%d] to compute chan max ..." % (xmin,xmax,ymin,ymax))
			data_ch= data[ymin:ymax, xmin:xmax, self.chref]
		
		cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))		
		data_masked= np.ma.masked_where(~cond_ch, data_ch, copy=False)
		data_max= data_masked.max()
		logger.debug("Chan %d max: %s" % (self.chref, str(data_max)))
	
		# - Check that channels are not entirely negatives
		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			if self.use_mask_box:
				data_ch= data[ymin:ymax, xmin:xmax,i]
			cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond_ch]
			data_ch_max= data_ch_1d.max()
			if data_ch_max<=0 or not np.isfinite(data_ch_max):
				logger.warn("Chan %d max is <=0 or not finite, returning None!" % (i))
				return None

		# - Scale data
		data_scaled= data/data_max
		data_scaled[~cond]= 0 # Restore 0 and nans set in original data

		return data_scaled

##############################
##   MinShifter
##############################
class MinShifter(object):
	""" Shift data to min, e.g. subtract min from each pixel """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.chid= -1 # do for all channels, otherwise on just selected channel
		if 'chid' in kwparams:	
			self.chid= kwparams['chid']
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and shift
		data_shifted= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond]
			data_ch_min= data_ch_1d.min()
			data_ch_shifted= (data_ch-data_ch_min)
			data_ch_shifted[~cond]= 0 # Set 0 and nans in original data to min
			data_shifted[:,:,i]= data_ch_shifted

		return data_shifted


##############################
##   Shifter
##############################
class Shifter(object):
	""" Shift data to input value """

	def __init__(self, offsets, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.offsets= offsets
		
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check size of offsets
		nchannels= data.shape[2]
		noffsets= len(self.offsets)
		if noffsets<=0 or noffsets!=nchannels:
			logger.error("Empty offsets or size different from data channels!")
			return None

		# - Shift data
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_shifted= (data-self.offsets)
		data_shifted[~cond]= 0

		return data_shifted


##############################
##   Standardizer
##############################
class Standardizer(object):
	""" Standardize data according to given means and sigmas """

	def __init__(self, means, sigmas, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.means= means
		self.sigmas= sigmas

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check size of means/sigmas
		nchannels= data.shape[2]
		nmeans= len(self.means)
		if nmeans<=0 or nmeans!=nchannels:
			logger.error("Empty means or size different from data channels!")
			return None
		nsigmas= len(self.sigmas)
		if nsigmas<=0 or nsigmas!=nchannels:
			logger.error("Empty sigmas or size different from data channels!")
			return None

		# - Transform data
		cond= np.logical_and(data!=0, np.isfinite(data))
		data_norm= (data-self.means)/self.sigmas
		data_norm[~cond]= 0

		return data_norm

##############################
##   NegativeDataFixer
##############################
class NegativeDataFixer(object):
	""" Shift data to min for entirely negative channels """

	def __init__(self, **kwparams):
		""" Create a data pre-processor object """

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Find negative channels
		data_shifted= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_ch_1d= data_ch[cond]
			data_ch_min= data_ch_1d.min()
			data_ch_max= data_ch_1d.max()

			if data_ch_max>0:
				continue

			data_ch_shifted= (data_ch-data_ch_min)
			data_ch_shifted[~cond]= 0 # Set 0 and nans in original data to min
			data_shifted[:,:,i]= data_ch_shifted
			

		return data_shifted

		
##############################
##   Scaler
##############################
class Scaler(object):
	""" Scale data by a factor """

	def __init__(self, scale_factors, **kwparams):
		""" Create a data pre-processor object """
	
		# - Set parameters
		self.scale_factors= self.scale_factors
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check size of scale factors
		nchannels= data.shape[2]
		nscales= len(self.scale_factors)
		if nscales<=0 or nscales!=nchannels:
			logger.error("Empty scale factors or size different from data channels!")
			return None

		# - Apply scale factors
		data_scaled= data*self.scale_factors

		return data_scaled


##############################
##   LogStretcher
##############################
class LogStretcher(object):
	""" Apply log transform to data """

	def __init__(self, chid=-1, minmaxnorm=False, data_norm_min=-6, data_norm_max=6, clip_neg=False, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.chid= chid # do for all channels, otherwise skip selected channel
		self.minmaxnorm= minmaxnorm
		self.data_norm_min= data_norm_min
		self.data_norm_max= data_norm_max
		self.clip_neg= clip_neg

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channel and convert to lg
		data_transf= np.copy(data)

		for i in range(data.shape[-1]):
			# - Exclude channel?
			if self.chid!=-1 and i==self.chid:
				continue

			data_ch= data[:,:,i]
			badpix_cond= np.logical_or(data_ch==0, ~np.isfinite(data_ch))
			cond_ch= np.logical_and(data_ch>0, np.isfinite(data_ch))

			# - Check that there are pixel >0 for log transform
			data_ch_1d= data_ch[cond_ch]
			if data_ch_1d.size<=0:
				logger.warn("All pixels in channel %d are negative and cannot be log transformed, returning None!" % (i))
				return None

			# - Apply log
			data_ch_lg= np.log10(data_ch, where=cond_ch)
			data_ch_lg_1d= data_ch_lg[cond_ch]
			data_ch_lg_min= data_ch_lg_1d.min()
			##data_ch_lg[~cond_ch]= 0
			data_ch_lg[~cond_ch]= data_ch_lg_min

			# - Apply min/max norm data using input parameters
			if self.minmaxnorm:
				data_ch_lg_norm= (data_ch_lg-self.data_norm_min)/(self.data_norm_max-self.data_norm_min)
				if self.clip_neg:
					data_ch_lg_norm[data_ch_lg_norm<0]= 0
				data_ch_lg= data_ch_lg_norm
				#data_ch_lg[~cond_ch]= 0
				data_ch_lg[badpix_cond]= 0
				
			# - Set in cube
			data_transf[:,:,i]= data_ch_lg


		return data_transf

##############################
##   BorderMasker
##############################
class BorderMasker(object):
	""" Mask input data at borders """

	def __init__(self, mask_fract=0.7, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.mask_fract= mask_fract

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Mask all channels at border
		logger.debug("Masking all channels at border (fract=%f) ..." % (self.mask_fract))
		data_masked= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data[:,:,i]
			cond= np.logical_and(data_ch!=0, np.isfinite(data_ch))
			data_shape= data_ch.shape
			data_ch_1d= data_ch[cond]
			data_min= data_ch_1d.min()
			mask= np.zeros(data_shape)
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.debug("Masking chan %d (%d,%d) in range x[%d,%d] y[%d,%d]" % (i, data_shape[0], data_shape[1], xmin, xmax, ymin, ymax))
			mask[ymin:ymax, xmin:xmax]= 1
			data_ch[mask==0]= 0
			##data_ch[mask==0]= data_min
			data_masked[:,:,i]= data_ch
	
		return data_masked

##############################
##   BkgSubtractor
##############################
class BkgSubtractor(object):
	""" Subtract background from input data """

	def __init__(self, sigma=3, use_mask_box=False, mask_fract=0.7, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma= sigma
		self.use_mask_box= use_mask_box
		self.mask_fract= mask_fract
		self.chid= chid # -1=do for all channels, otherwise subtract only from selected channel

	def __subtract_bkg(self, data):
		""" Subtract background from channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		
		# - Mask region at image center (where source is supposed to be)?
		bkgdata= np.copy(data) 
		if self.use_mask_box:
			data_shape= data.shape
			xc= int(data_shape[1]/2)
			yc= int(data_shape[0]/2)
			dy= int(data_shape[0]*self.mask_fract/2.)
			dx= int(data_shape[1]*self.mask_fract/2.)
			xmin= xc - dx
			xmax= xc + dx
			ymin= yc - dy
			ymax= yc + dy
			logger.debug("Masking data (%d,%d) in range x[%d,%d] y[%d,%d]" % (data_shape[0], data_shape[1], xmin, xmax, ymin, ymax))
			bkgdata[ymin:ymax, xmin:xmax]= 0
	
		# - Compute and subtract mean bkg from data
		logger.debug("Subtracting bkg ...")
		cond_bkg= np.logical_and(bkgdata!=0, np.isfinite(bkgdata))
		bkgdata_1d= bkgdata[cond_bkg]
		logger.debug("--> bkgdata min/max=%s/%s" % (str(bkgdata_1d.min()), str(bkgdata_1d.max())))

		bkgval, _, _ = sigma_clipped_stats(bkgdata_1d, sigma=self.sigma)

		data_bkgsub= data - bkgval
		data_bkgsub[~cond]= 0
		cond_bkgsub= np.logical_and(data_bkgsub!=0, np.isfinite(data_bkgsub))
		data_bkgsub_1d= data_bkgsub[cond_bkgsub]

		logger.debug("--> data min/max (after bkgsub)=%s/%s (bkg=%s)" % (str(data_bkgsub_1d.min()), str(data_bkgsub_1d.max()), str(bkgval)))

		return data_bkgsub


	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and get bgsub data
		data_bkgsub= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue	
			data_ch_bkgsub= self.__subtract_bkg(data[:,:,i])
			data_bkgsub[:,:,i]= data_ch_bkgsub

		return data_bkgsub


##############################
##   SigmaClipShifter
##############################
class SigmaClipShifter(object):
	""" Shift all pixels to new zero value equal to mean+(sigma*std) and clip values below this zero """

	def __init__(self, sigma=1.0, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma= sigma
		self.chid= chid # -1=do for all channels, otherwise clip only selected channel

	def __clip(self, data):
		""" Clip channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip (considered noise)
		#   NB: Following Galvin et al, PASA 131, 1 (2019)
		logger.debug("Clipping all pixels below (mean + %f x stddev) ..." % (self.sigma))
		clipmean, median, stddev = sigma_clipped_stats(data_1d, sigma=self.sigma)

		newzero= clipmean + self.sigma*stddev

		data_clipped= np.copy(data)
		#data_clipped[data_clipped<clipmean]= clipmean #### CHECK!!! PROBABLY WRONG!!!
		data_clipped-= newzero
		data_clipped[data_clipped<0]= 0
		data_clipped[~cond]= 0
		cond_clipped= np.logical_and(data_clipped!=0, np.isfinite(data_clipped))
		data_clipped_1d= data_clipped[cond_clipped]

		logger.debug("--> data min/max (after sigmaclip)=%s/%s (clipmean=%s)" % (str(data_clipped_1d.min()), str(data_clipped_1d.max()), str(clipmean)))

		return data_clipped 
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and get bgsub data
		data_clipped= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue	
			data_ch_clipped= self.__clip(data[:,:,i])
			data_clipped[:,:,i]= data_ch_clipped

		return data_clipped


##############################
##   SigmaClipper
##############################
class SigmaClipper(object):
	""" Clip all pixels below zlow=mean-(sigma_low*std) and above zhigh=mean + (sigma_up*std) """

	def __init__(self, sigma_low=10.0, sigma_up=10.0, chid=-1, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.sigma_low= sigma_low
		self.sigma_up= sigma_up
		self.chid= chid # -1=do for all channels, otherwise clip only selected channel

	def __clip(self, data):
		""" Clip channel input """

		cond= np.logical_and(data!=0, np.isfinite(data))
		data_1d= data[cond]

		# - Clip all pixels that are below sigma clip
		logger.debug("Clipping all pixel values <(mean - %f x stddev) and >(mean + %f x stddev) ..." % (self.sigma_low, self.sigma_up))
		res= sigma_clip(data_1d, sigma_lower=self.sigma_low, sigma_upper=self.sigma_up, masked=True, return_bounds=True)
		thr_low= res[1]
		thr_up= res[2]

		data_clipped= np.copy(data)
		data_clipped[data_clipped<thr_low]= thr_low
		data_clipped[data_clipped>thr_up]= thr_up
		data_clipped[~cond]= 0
		
		return data_clipped 
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Loop over channels and get bgsub data
		data_clipped= np.copy(data)

		for i in range(data.shape[-1]):
			if self.chid!=-1 and i!=self.chid:
				continue	
			data_ch_clipped= self.__clip(data[:,:,i])
			data_clipped[:,:,i]= data_ch_clipped

		return data_clipped

##############################
##   Resizer
##############################
class Resizer(object):
	""" Resize image to desired size """

	def __init__(self, resize_size, preserve_range=True, upscale=False, downscale_with_antialiasing=False, set_pad_val_to_min=True, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.resize_size= resize_size
		self.preserve_range= preserve_range
		self.upscale= upscale # Upscale images to resize size when original image size is smaller than desired size. If false, pad to reach desired size
		self.downscale_with_antialiasing=downscale_with_antialiasing  # Use antialiasing when down-scaling an image
		self.set_pad_val_to_min= set_pad_val_to_min
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check if resizing is needed
		data_shape= data.shape
		nx= data_shape[1]
		ny= data_shape[0]
		nchannels= data_shape[2]
		is_same_size= (nx==self.resize_size) and (ny==self.resize_size)
		if is_same_size:
			logger.debug("Images have already the desired size (%d,%d), nothing to be done..." % (ny, nx))
			return data

		# - Select resizing options
		max_dim= self.resize_size
		min_dim= self.resize_size
		if not self.upscale:
			min_dim=None

		downscaling= (nx>self.resize_size) and (ny>self.resize_size)
		antialiasing= False
		if downscaling and self.downscale_with_antialiasing:
			antialiasing= True

		interp_order= 1 # 1=bilinear, 2=biquadratic, 3=bicubic, 4=biquartic, 5=biquintic

		# - Resize data
		try:
			try: # work for skimage<=0.15.0
				ret= resize_img_v2(data, 
					min_dim=min_dim, max_dim=max_dim, min_scale=None, mode="square", 
					order=interp_order, anti_aliasing=antialiasing, 
					preserve_range=self.preserve_range
				)
			except:
				ret= resize_img_v2(img_as_float64(data), 
					min_dim=min_dim, max_dim=max_dim, min_scale=None, mode="square", 
					order=interp_order, anti_aliasing=antialiasing, 
					preserve_range=self.preserve_range
				)

			data_resized= ret[0]
			#window= ret[1]
			#scale= ret[2] 
			#padding= ret[3] 
			#crop= ret[4]

		except Exception as e:
			logger.warn("Failed to resize data to size (%d,%d) (err=%s)!" % (self.resize_size, self.resize_size, str(e)))
			return None

		if data_resized is None:
			logger.warn("Resized data is None, failed to resize to size (%d,%d) (see logs)!" % (self.resize_size, self.resize_size))

		if self.set_pad_val_to_min:
			for i in range(data_resized.shape[-1]):
				data_ch= data_resized[:,:,i]
				cond_ch= np.logical_and(data_ch!=0, np.isfinite(data_ch))
				data_ch_1d= data_ch[cond_ch]
				data_min= data_ch_1d.min()
				data_ch[~cond_ch]= data_min
				data_resized[:,:,i]= data_ch

		return data_resized



##############################
##   ChanDivider
##############################
class ChanDivider(object):
	""" Divide channel by reference channel """

	def __init__(self, chref=0, logtransf=False, strip_chref=False, trim=False, trim_min=-6, trim_max=6, **kwparams):
		""" Create a data pre-processor object """

		# - Set parameters
		self.chref= chref
		self.logtransf= logtransf
		self.strip_chref= strip_chref
		self.trim= trim
		self.trim_min= trim_min
		self.trim_max= trim_max
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """
			
		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Init ref channel
		cond= np.logical_and(data!=0, np.isfinite(data)) 
		data_ref= np.copy(data[:,:,self.chref])
		cond_ref= np.logical_and(data_ref!=0, np.isfinite(data_ref))

		# - Divide other channels by reference channel
		data_norm= np.copy(data)
		data_denom= np.copy(data_ref)
		data_denom[data_denom==0]= 1

		for i in range(data_norm.shape[-1]):
			if i==self.chref:
				data_norm[:,:,i]= np.copy(data_ref)
			else:
				logger.debug("Divide channel %d by reference channel %d ..." % (i, self.chref))
				dn= data_norm[:,:,i]/data_denom
				dn[~cond_ref]= 0 # set ratio to zero if ref pixel flux was zero or nan
				data_norm[:,:,i]= dn

		data_norm[~cond]= 0

		# - Apply log transform to ratio channels?
		if self.logtransf:
			logger.debug("Applying log-transform to channel ratios ...")
			data_transf= np.copy(data_norm)
			data_transf[data_transf<=0]= 1
			data_transf_lg= np.log10(data_transf)
			data_transf= data_transf_lg
			data_transf[~cond]= 0

			if self.trim:
				data_transf[data_transf>self.trim_max]= self.trim_max
				data_transf[data_transf<self.trim_min]= self.trim_min

			data_transf[:,:,self.chref]= data_norm[:,:,self.chref]
			data_norm= data_transf

		# - Strip ref channel 
		if self.strip_chref:
			data_norm_striprefch= np.delete(data_norm, chref, axis=2)
			data_norm= data_norm_striprefch
			
		return data_norm


##############################
##   ZScaleTransformer
##############################
class ZScaleTransformer(object):
	""" Apply zscale transformation to each channel """

	def __init__(self, contrasts=[0.25,0.25,0.25], **kwparams):
		""" Create a data pre-processor object """

		self.contrasts= contrasts
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Check constrast dim vs nchans
		nchans= data.shape[-1]
	
		if len(self.contrasts)<nchans:
			logger.error("Invalid constrasts given (constrast list size=%d < nchans=%d)" % (len(self.contrasts), nchans))
			return None
		
		# - Transform each channel
		data_stretched= np.copy(data)

		for i in range(data.shape[-1]):
			data_ch= data_stretched[:,:,i]
			transform= ZScaleInterval(contrast=self.contrasts[i]) # able to handle NANs
			data_transf= transform(data_ch)
			data_stretched[:,:,i]= data_transf

		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched


##############################
##   HistEqualizer
##############################
class HistEqualizer(object):
	""" Apply histogram equalization to each channel """

	def __init__(self, adaptive=False, clip_limit=0.03, **kwparams):
		""" Create a data pre-processor object """

		self.clip_limit= clip_limit
		self.adaptive= adaptive

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))
		nchans= data.shape[-1]
	
		# - Transform each channel
		data_stretched= np.copy(data)

		for i in range(data.shape[-1]):
			if self.adaptive:
				data_stretched[:,:,i]= equalize_adapthist(data[:,:,i], clip_limit=self.clip_limit)
			else:
				data_stretched[:,:,i]= equalize_hist(data[:,:,i])
				#interval = MinMaxInterval()
				#transform= HistEqStretch( interval(data[:,:,i]) ) # astropy implementation
				#data_stretched[:,:,i]= transform( interval(data[:,:,i]) )
				
		# - Scale data
		data_stretched[~cond]= 0 # Restore 0 and nans set in original data

		return data_stretched




##############################
##   Chan3Trasformer
##############################
class Chan3Trasformer(object):
	""" Create 3 channels with a different transform per each channel: 
				- ch1: sigmaclip(0,20) + zscale(0.25)
				- ch2: sigmaclip(1,20) + zscale(0.25)
				- ch3: histeq
	"""

	def __init__(self, sigma_clip_baseline=0, sigma_clip_low=1, sigma_clip_up=20, zscale_contrast=0.25, **kwparams):
		""" Create a data pre-processor object """

		self.sigma_clip_baseline= sigma_clip_baseline
		self.sigma_clip_low= sigma_clip_low
		self.sigma_clip_up= sigma_clip_up
		self.zscale_contrast= zscale_contrast
		
	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))
		nchans= data.shape[-1]
	
		# - Apply ChanResizer to create 3 channels
		cr= ChanResizer(nchans=3)
		data_cube= cr(data)

		# - Define transforms
		sclipper= SigmaClipper(sigma_low=self.sigma_clip_baseline, sigma_up=self.sigma_clip_up, chid=-1)
		sclipper2= SigmaClipper(sigma_low=self.sigma_clip_low, sigma_up=self.sigma_clip_up, chid=-1)
		zscale= ZScaleTransformer(contrasts=[self.zscale_contrast])
		histEq= HistEqualizer(adaptive=False)
		#sremover= SourceRemover(niters=self.niters, seed_thr=self.seed_thr, npix_max_thr=self.npix_max_thr)

		# - Create channel 1: sigmaclip(0,20) + zscale(0.25)	
		data_transf_ch1= zscale(sclipper(np.expand_dims(data_cube[:,:,0], axis=-1)))
		data_cube[:,:,0]= data_transf_ch1[:,:,0]

		# - Create channel 2: sigmaclip(1,20) + zscale(0.25)
		data_transf_ch2= zscale(sclipper2(np.expand_dims(data_cube[:,:,1], axis=-1)))
		data_cube[:,:,1]= data_transf_ch2[:,:,0]

		# - Create channel 3: histeq
		#data_transf_ch3= histEq(zscale(np.expand_dims(data_cube[:,:,2], axis=-1)))
		#data_transf_ch3= histEq(sremover(np.expand_dims(data_cube[:,:,2], axis=-1)))
		#data_transf_ch3= histEq(sremover(sclipper(np.expand_dims(data_cube[:,:,2], axis=-1))))
		data_transf_ch3= histEq( np.expand_dims(data_cube[:,:,2], axis=-1) )
		data_cube[:,:,2]= data_transf_ch3[:,:,0]
		
		return data_cube

##############################
##   ChanResizer
##############################
class ChanResizer(object):
	""" ChanResizer modifies the number of channels until reaching desidered value. Replicate last channel when expanding. """

	def __init__(self, nchans, **kwparams):
		""" Create a data pre-processor object """

		self.nchans= nchans
		self.nchans_max= 1000
		

	def __call__(self, data):
		""" Apply transformation and return transformed data """

		# - Check data
		if data is None:
			logger.error("Input data is None!")
			return None

		# - Check number of given chans		
		if self.nchans>self.nchans_max or self.nchans<=0:
			logger.error("Invalid channel specified or too many channels desired (%d) (hint: the maximum is %d)!" % (self.nchans, self.nchans_max))
			return None

		cond= np.logical_and(data!=0, np.isfinite(data))

		# - Set nchan curr 
		ndim_curr= data.ndim
		if ndim_curr==2:
			nchans_curr= 1
		else:
			nchans_curr= data.shape[-1]
		
		if self.nchans==nchans_curr:
			logger.debug("Desired number of channels equal to current, nothing to be done...")
			return data
		
		expanding= self.nchans>nchans_curr

		# - Expand array first?
		#   NB: If 2D first create an extra dimension
		if ndim_curr==2:
			data= np.expand_dims(data, axis=-1)

		# - Copy last channel in new ones
		data_resized= np.zeros((data.shape[0], data.shape[1], self.nchans))

		if expanding:
			for i in range(self.nchans):
				if i<nchans_curr:
					data_resized[:,:,i]= data[:,:,i]
				else:
					data_resized[:,:,i]= data[:,:,nchans_curr-1]	
		else:
			for i in range(self.nchans):
				data_resized[:,:,i]= data[:,:,i]

		return data_resized




