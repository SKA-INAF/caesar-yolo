#!/bin/bash

INPUTFILE="galaxy0001.fits"
WEIGHTFILE="../share/weights-yolov8l_scratch_imgsize640_nepochs300.pt"
IMGSIZE=640
PREPROC_OPTS="--imgsize=$IMGSIZE --preprocessing --zscale_stretch --zscale_contrasts=0.25,0.25,0.25 --normalize_minmax --norm_min=0 --norm_max=255 "
SCORE_THR=0.5
IOU_THR_SOFT=0.3
IOU_THR_HARD=0.8
DET_OPTS="--scoreThr=$SCORE_THR --merge_overlap_iou_thr_soft=$IOU_THR_SOFT --merge_overlap_iou_thr_hard=$IOU_THR_HARD "
DRAW_OPTS="--draw_plots --save_plots --draw_class_label_in_caption "

python ../scripts/run.py --image=$INPUTFILE --weights=$WEIGHTFILE \
  $PREPROC_OPTS \
  $DET_OPTS \
  $DRAW_OPTS

