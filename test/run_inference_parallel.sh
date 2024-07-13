#!/bin/bash

#export HWLOC_COMPONENTS=-gl

NPROC=4

INPUTFILE="cutout_G005.5+0.0IFx_Mosaic_Mom0.fits"
WEIGHTFILE="../share/weights-yolov8l_scratch_imgsize640_nepochs300.pt"
IMGSIZE=640
PREPROC_OPTS="--imgsize=$IMGSIZE --preprocessing --zscale_stretch --zscale_contrasts=0.25,0.25,0.25 --normalize_minmax --norm_min=0 --norm_max=255 "
SCORE_THR=0.5
IOU_THR_SOFT=0.3
IOU_THR_HARD=0.8
DET_OPTS="--scoreThr=$SCORE_THR --merge_overlap_iou_thr_soft=$IOU_THR_SOFT --merge_overlap_iou_thr_hard=$IOU_THR_HARD "
DRAW_OPTS="--draw_plots --save_plots --draw_class_label_in_caption "
PARALLEL_OPTS="--split_img_in_tiles --tile_xsize=256 --tile_ysize=256 --tile_xstep=1 --tile_ystep=1 "
##SAVE_OPTS="--save_tile_catalog --save_tile_region --save_tile_img " # TO SAVE SUBTILE CATALOG OUTPUTS (DEBUG)
SAVE_OPTS=" " # ONLY SAVE FINAL CATALOGUE

mpirun -np $NPROC python ../scripts/run.py --image=$INPUTFILE --weights=$WEIGHTFILE \
  $PREPROC_OPTS \
  $DET_OPTS \
  $DRAW_OPTS \
  $PARALLEL_OPTS \
  $SAVE_OPTS
