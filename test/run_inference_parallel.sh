#!/bin/bash

#######################
##   SET OPTIONS
#######################
#export HWLOC_COMPONENTS=-gl

# - INPUT OPTIONS
INPUTFILE="cutout_G005.5+0.0IFx_Mosaic_Mom0.fits"
WEIGHTFILE="../share/weights-yolov8l_scratch_imgsize640_nepochs300.pt"

# - RUN OPTIONS
DEVICES="cpu"
##DEVICES="cuda:0"
##DEVICES="cuda:0,cuda:1,cuda:2,cuda:3"
MPI_NPROC=4
MAX_NTASKS_PER_WORKER=1000

MPI_NPROC=4
TILE_SIZE=256
TILE_STEP=1

##RUN_OPTS="--devices=$DEVICES --multigpu --max_ntasks_per_worker=$MAX_NTASKS_PER_WORKER --split_img_in_tiles --tile_xsize=$TILE_SIZE --tile_ysize=$TILE_SIZE --tile_xstep=$TILE_STEP --tile_ystep=$TILE_STEP "
RUN_OPTS="--devices=$DEVICES --max_ntasks_per_worker=$MAX_NTASKS_PER_WORKER --split_img_in_tiles --tile_xsize=$TILE_SIZE --tile_ysize=$TILE_SIZE --tile_xstep=$TILE_STEP --tile_ystep=$TILE_STEP "

# - PREPROCESSING OPTIONS
IMGSIZE=640
PREPROC_OPTS="--preprocessing --imgsize=$IMGSIZE --zscale_stretch --zscale_contrasts=0.25,0.25,0.25 --normalize_minmax --norm_min=0 --norm_max=255 "

# - INFERENCE OPTIONS
SCORE_THR=0.5
IOU_THR_SOFT=0.3
IOU_THR_HARD=0.8
DET_OPTS="--scoreThr=$SCORE_THR --merge_overlap_iou_thr_soft=$IOU_THR_SOFT --merge_overlap_iou_thr_hard=$IOU_THR_HARD "
DRAW_OPTS="--draw_plots --save_plots --draw_class_label_in_caption "

# - SAVE OPTIONS
SAVE_OPTS=""
##SAVE_OPTS="--save_tile_catalog --save_tile_region --save_tile_img "

#########################
##   RUN
#########################
echo "INFO: Starting inference run ..."
date

mpirun -np $MPI_NPROC python ../scripts/run.py --image=$INPUTFILE --weights=$WEIGHTFILE \
  $PREPROC_OPTS \
  $DET_OPTS \
  $DRAW_OPTS \
  $RUN_OPTS \
  $SAVE_OPTS

date
echo "INFO: Inference run completed."
