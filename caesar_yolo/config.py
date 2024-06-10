#CLASS_DICT = {'background': 0, 'spurious': 1, 'compact': 2, 'extended': 3, 'extended-multisland': 4, 'diffuse': 5}
#CLASSES_NUM = len(CLASS_DICT.keys())

CONFIG = {

	#######################################
	##    ADDON INFERENCE OPTIONS
	#######################################
	# - Image resize
	'img_size': 640,
	
	# - Preprocessor function
	'preprocess_fcn': None,

	# - Image read options
	'image_path': '',
	'image_xmin': 0,
	'image_xmax': 0,
	'image_ymin': 0,
	'image_ymax': 0,
		
	# - Image parallel read options
	'mpi': None,
	'split_image_in_tiles': False,
	'tile_xsize': 256, # in pixels
	'tile_ysize': 256, # in pixels
	'tile_xstep': 1.0, # [0,1], 1=no overlap
	'tile_ystep': 1.0, # [0,1], 1=no overlap
	'max_ntasks_per_worker': 100,

	# - Source detection options
	#'iou_thr': 0.6,
	'merge_overlap_iou_thr_soft': 0.3,
	'merge_overlap_iou_thr_hard': 0.8,
	'score_thr': 0.7,

	# - Catalog output file options
	'save_catalog': True,
	'save_tile_catalog': False,
	'save_region': True,
	'save_tile_region': False,
	'outfile': '',
	'outfile_json': '',
	
	# - Save inference plot
	'draw_plot': False,
	'draw_class_label_in_caption': True,
	'save_plot': False,
	

}## close dict
