import os
import sys
from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Undirected graph
class Graph:

	# Init function to declare class variables
	def __init__(self,V):
		self.V = V
		self.adj = [[] for i in range(V)]

	def DFSUtil(self, temp, v, visited):

		# Mark the current vertex as visited
		visited[v] = True

		# Store the vertex to list
		temp.append(v)

		# Repeat for all vertices adjacent to this vertex v
		for i in self.adj[v]:
			if visited[i] == False:
 				# Update the list
 				temp = self.DFSUtil(temp, i, visited)

		return temp

	# method to add an undirected edge
	def addEdge(self, v, w):
		self.adj[v].append(w)
		self.adj[w].append(v)

	# Method to retrieve connected components in an undirected graph
	def connectedComponents(self):
		visited = []
		cc = []
		for i in range(self.V):
			visited.append(False)
		for v in range(self.V):
			if visited[v] == False:
				temp = []
				cc.append(self.DFSUtil(temp, v, visited))

		return cc


def get_iou(bb1, bb2):
	""" Calculate the Intersection over Union (IoU) of two bounding boxes """

	bb1_x1= bb1[0]
	bb1_y1= bb1[1]
	bb1_x2= bb1[2]
	bb1_y2= bb1[3]

	bb2_x1= bb2[0]
	bb2_y1= bb2[1]
	bb2_x2= bb2[2]
	bb2_y2= bb2[3]

	#bb1_x1= bb1[1]
	#bb1_y1= bb1[0]
	#bb1_x2= bb1[3]
	#bb1_y2= bb1[2]

	#bb2_x1= bb2[1]
	#bb2_y1= bb2[0]

	#bb2_x2= bb2[3]
	#bb2_y2= bb2[2]

	assert bb1_x1 < bb1_x2
	assert bb1_y1 < bb1_y2
	assert bb2_x1 < bb2_x2
	assert bb2_y1 < bb2_y2

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1_x1, bb2_x1)
	y_top = max(bb1_y1, bb2_y1)
	x_right = min(bb1_x2, bb2_x2)
	y_bottom = min(bb1_y2, bb2_y2)

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
	bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0

	return iou
    
    
def predict(
		model, 
		imgpath, 
		score_thr=0.5,
		merge_overlap_iou_thr_soft= 0.3,
		merge_overlap_iou_thr_hard=0.8,
		gtdata=None,
		draw=False
	):
	""" Predict on image """

	# - Init options
	class_color_map= {
		'bkg': (0,0,0),# black
		'spurious': (1,0,0),# red
		'compact': (0,0,1),# blue
		'extended': (1,1,0),# green	
		'extended-multisland': (1,0.647,0),# orange
		'flagged': (0,0,0),# black
		}
	draw_class_label_in_caption= False

	# - Read image
	img= plt.imread(imgpath)
	
	# - Compute predictions
	results= model(
		imgpath, 
		save=False, 
		imgsz=640, 
		conf=score_thr, 
		iou=0.1,
		visualize=False,
		show=False,
		show_labels=False,
		show_conf=False,
		show_boxes=False
	)
	
	# - Loop through predictions and select bboxes > scores
	bboxes_det= []
	scores_det= []
	labels_det= []
	
  
	for result in results:
		bboxes= result.boxes.xyxy.cpu().numpy()   # box with xywh format, (N, 4)
		scores= result.boxes.conf.cpu().numpy()   # confidence score, (N, 1)
		cls= result.boxes.cls.cpu().numpy()    # cls, (N, 1)
		class_labels= [model.names[int(item)] for item in cls]
 	 	
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
			if score<score_thr:
				continue
			scores_det.append(score)
			bboxes_det.append(bbox)
			labels_det.append(label)
  		
  # - Find overlapped bboxes with same labels, keep only that with higher confidence
	N= len(bboxes_det)
	g= Graph(N)
	for i in range(N-1):
		#x1= bboxes_det[i][0]
		#y1= bboxes_det[i][1]
		#x2= bboxes_det[i][2]
		#y2= bboxes_det[i][3]
		
		for j in range(i+1,N):
			#x1_j= bboxes_det[j][0]
			#y1_j= bboxes_det[j][1]
			#x2_j= bboxes_det[j][2]
			#y2_j= bboxes_det[j][3]
		
			same_class= (labels_det[i]==labels_det[j])
			
			#iou= get_iou((y1,x1,y2,x2), (y1_j,x1_j,y2_j,x2))
			iou= get_iou(bboxes_det[i], bboxes_det[j])
			overlapping_soft= (iou>=merge_overlap_iou_thr_soft)
			overlapping_hard= (iou>=merge_overlap_iou_thr_hard)
			mergeable= (overlapping_hard or (same_class and overlapping_soft))
			print("IoU(%d,%d)=%f, mergeable? %d" % (i+1, j+1, iou, mergeable))
  
			if mergeable:
				g.addEdge(i,j)
  
  # - Select connected boxes
	cc = g.connectedComponents()
	bboxes_sel= []
	scores_sel= []
	labels_sel= []
  
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
		
		print("Obj no. %d: bbox(%s), score=%f, label=%s" % (i+1, str(bboxes_det[index_best]), scores_det[index_best], labels_det[index_best]))
		
	print("#%d selected objects left after merging overlapping ones ..." % len(bboxes_sel))
			
	# - Draw?
	if draw:
		figsize=(16,16)
		fig, ax = plt.subplots(1, figsize=figsize)
		#fig, ax = plt.subplots()
		
		# - Show area outside image boundaries
		title= imgpath
		height, width = img.shape[:2]
		
		#ax.set_ylim(height + 10, -10)
		#ax.set_xlim(-10, width + 10)
		ax.set_ylim(height + 2, -2)
		ax.set_xlim(-2, width + 2)
		ax.axis('off')
		
		ax.imshow(img)

		# - Draw GT bounding box rect?
		if gtdata is not None and gtdata["bboxes"]:
			bboxes_gt= gtdata["bboxes"]
			labels_gt= gtdata["labels"]
			for i in range(len(bboxes_gt)):
				label= labels_gt[i]
				color_gt= class_color_map[label]
				x1, y1, x2, y2 = bboxes_gt[i]
				p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, alpha=0.7, linestyle="dashed", edgecolor=color_gt, facecolor='none')
				ax.add_patch(p)
				caption = ""
				ax.text(x1, y1 + 8, caption, color='w', size=13, backgroundcolor="none")

		for i in range(len(bboxes_sel)):
			bbox= bboxes_sel[i]
			score= scores_sel[i]
			label= labels_sel[i]
			color = class_color_map[label]
			
			
			# - Draw bounding box rect
			x1= bbox[0]
			y1= bbox[1]
			x2= bbox[2]
			y2= bbox[3]
			dx= x2-x1
			dy= y2-y1
			
			rect= patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, alpha=0.7, linestyle="solid", edgecolor=color, facecolor='none')
			#rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')	
			ax.add_patch(rect)
			
			# Label
			if draw_class_label_in_caption:
				caption = "{} {:.2f}".format(label, score)
				ax.text(x1, y1 + 8, caption, color=color, size=20, backgroundcolor="none")
			else:
				caption = "{:.2f}".format(score)
				#ax.text(x1 + dx/2 - 4, y1 - 1, caption, color=color, size=23, backgroundcolor="none")
				#ax.text(x1 + dx/2 - 4, y1 - 1, caption, color="mediumturquoise", size=23, backgroundcolor="none")
				ax.text(x1 + dx/2 - 4, y1 - 1, caption, color="darkturquoise", size=30, backgroundcolor="none")

			
		plt.show()
		
	#model.predict(
	#	imgpath, 
	#	save=False, 
	#	imgsz=640, 
	#	conf=0.25, 
	#	iou=0.1,
	#	visualize=False,
	#	show=True,
	#	show_labels=True,
	#	show_conf=True,
	#	show_boxes=True
	#)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	return (bboxes_sel, labels_sel, scores_sel)


def read_filelist(filename):
	""" Read filelist """
	
	with open(filename) as fp:
		filenames= fp.readlines()
		
	return filenames

###############################
##   COMPLETENESS
###############################
def compute_completeness(gtdata_list, preddata_list):
	""" Compute completeness """
		
	# - Loop over GT data
	nSources= 0
	nSources_det= 0
	nSpuriousSources= 0
	nSpuriousSources_det= 0
	nFlaggedSources= 0
	nFlaggedSources_det= 0
	nSourcesPerClass= {
		"compact": 0,
		"extended": 0,
		"extended-multisland": 0
	}
	nSourcesPerClass_det= {
		"compact": 0,
		"extended": 0,
		"extended-multisland": 0
	}
	
	for i in range(len(gtdata_list)):
		gtdata= gtdata_list[i]
		preddata= preddata_list[i]

		# - Search for matches
		imgname= gtdata["img"]
		bboxes= gtdata["bboxes"]
		labels= gtdata["labels"]
		bboxes_pred= preddata["bboxes"]
		labels_pred= preddata["labels"]
		scores_pred= preddata["scores"]
	
		for j in range(len(bboxes)):
			bbox= bboxes[j]
			label= labels[j]
			is_source= (label=="compact" or label=="extended" or label=="extended-multisland")
			is_spurious= (label=="spurious")
			is_flagged= (label=="flagged")
			detected= False
			iou_match= 0
			index_match= -1
			
			if label=="flagged":
				print("img=%s: label=%s, bbox=%s" % (imgname, label, bbox))
			
		
			for k in range(len(bboxes_pred)):
				bbox_pred= bboxes_pred[k]
				label_pred= labels_pred[k]
				iou= get_iou(bbox, bbox_pred)
				if iou>=iou_thr:
					detected= True
					if iou>iou_match:
						iou_match= iou
						index_match= k

				if label=="flagged":
					print("detobj no. %d, label=%s, bbox=%s, iou=%f" % (k+1, label_pred, bbox_pred, iou))
				

			label_det= "none"
			if detected:
				label_det= labels_pred[index_match]
			
			if label=="flagged":
				print("index_match=%d, label_det=%s, iou=%f" % (index_match, label_det, iou_match))
			
			is_source_det= (label_det=="compact" or label_det=="extended" or label_det=="extended-multisland")
			is_detected= (detected and is_source_det)
				
			#- Update source counts & metrics
			if is_spurious:
				nSpuriousSources+= 1
				if detected and label_det=="spurious":
					nSpuriousSources_det+= 1
		
			if is_flagged:
				nFlaggedSources+= 1
				if detected and label_det=="flagged": 
					nFlaggedSources_det+= 1
		
			if is_source:
				nSources+= 1
				nSourcesPerClass[label]+= 1

				if is_detected:
					nSources_det+= 1
					nSourcesPerClass_det[label]+= 1
	
	# - Compute cumulative completeness stats		
	completeness= float(nSources_det)/float(nSources)
	print("== COMPLETENESS ==")
	print("SOURCE (compact+extended+extended-multisland): n=%d, ndet=%d, C=%f" % (nSources, nSources_det, completeness))
	for cname in nSourcesPerClass:
		n= nSourcesPerClass[cname]
		ndet= nSourcesPerClass_det[cname]
		C= -999
		if n>0:
			C= float(ndet)/float(n)
	
		print("%s: n=%d, ndet=%d, C=%f" % (cname, n, ndet, C))
	
	C_spurious= 0
	if nSpuriousSources>0:
		C_spurious= float(nSpuriousSources_det)/float(nSpuriousSources)

	C_flagged= 0
	if nFlaggedSources>0:
		C_flagged= float(nFlaggedSources_det)/float(nFlaggedSources)	
	
	print("SPURIOUS: n=%d, ndet=%d, C=%f" % (nSpuriousSources, nSpuriousSources_det, C_spurious))
	print("FLAGGED: n=%d, ndet=%d, C=%f" % (nFlaggedSources, nFlaggedSources_det, C_flagged))
	print("==================")

###############################
##   RELIABILITY
###############################
def compute_reliability(gtdata_list, preddata_list):
	""" Compute reliability """
	
	# - Init counters
	nSources= 0
	nSources_matchingToGT= 0
	nSpuriousSources= 0
	nSpuriousSources_matchingToGT= 0
	nFlaggedSources= 0
	nFlaggedSources_matchingToGT= 0

	nSourcesPerClass= {
		"compact": 0,
		"extended": 0,
		"extended-multisland": 0
	}
	nSourcesPerClass_matchingToGT= {
		"compact": 0,
		"extended": 0,
		"extended-multisland": 0
	}
	
	for i in range(len(preddata_list)):
		gtdata= gtdata_list[i]
		preddata= preddata_list[i]

		# - Search for matches
		bboxes= gtdata["bboxes"]
		labels= gtdata["labels"]
		bboxes_pred= preddata["bboxes"]
		labels_pred= preddata["labels"]
		scores_pred= preddata["scores"]
	
		for j in range(len(bboxes_pred)):
			bbox_pred= bboxes_pred[j]
			label_pred= labels_pred[j]
			
			is_source_det= (label_pred=="compact" or label_pred=="extended" or label_pred=="extended-multisland")
			
			matching_gt= False
			iou_match= 0
			index_match= -1
		
			for k in range(len(bboxes)):
				bbox= bboxes[k]
				label= labels[k]
				iou= get_iou(bbox_pred, bbox)
				if iou>=iou_thr:
					matching_gt= True
					if iou>iou_match:
						iou_match= iou
						index_match= k
						
			label_gt= "none"
			if matching_gt:
				label_gt= labels[index_match]
				is_source= (label_gt=="compact" or label_gt=="extended" or label_gt=="extended-multisland");
			
			matching_to_true_source= (matching_gt and is_source)
							
			# - Update source counts & metrics
			if label_pred=="spurious":
				nSpuriousSources+= 1
				if matching_gt and label_gt=="spurious":
					nSpuriousSources_matchingToGT+= 1 
		
			if label_pred=="flagged":
				nFlaggedSources+= 1
				if matching_gt and label_gt=="flagged":
					nFlaggedSources_matchingToGT+= 1
		
			if is_source_det:
				nSources+= 1
				nSourcesPerClass[label_pred]+= 1
				
				if matching_to_true_source:
					nSources_matchingToGT+= 1
					nSourcesPerClass_matchingToGT[label_pred]+= 1
				
	# - Compute cumulative reliability
	reliability= float(nSources_matchingToGT)/float(nSources)
			
	print("== RELIABILITY ==")
	print("SOURCE (compact+extended+extended-multisland): ndet=%d, n_matchingToGT=%d, R=%f" % (nSources, nSources_matchingToGT, reliability))
	for cname_det in nSourcesPerClass:
		n_det= nSourcesPerClass[cname_det]
		n_matchingToGT= nSourcesPerClass_matchingToGT[cname_det]
		R= -999
		if n_det>0: 
			R= float(n_matchingToGT)/float(n_det)
		print("%s: n_det=%d, n_matchingToGT=%d, R=%f" % (cname_det, n_det, n_matchingToGT, R)) 
	
	R_spurious= 0
	R_flagged= 0
	if nSpuriousSources>0:
		R_spurious= float(nSpuriousSources_matchingToGT)/float(nSpuriousSources)
	if nFlaggedSources>0:
		R_flagged= float(nFlaggedSources_matchingToGT)/float(nFlaggedSources)
	
	print("SPURIOUS: n_det=%d, n_matchingToGT=%d, R=%f" % (nSpuriousSources, nSpuriousSources_matchingToGT, R_spurious))
	print("FLAGGED: n_det=%d, n_matchingToGT=%d, R=%f" % (nFlaggedSources, nFlaggedSources_matchingToGT, R_flagged)) 
	print("==================")
	

###########################
##    READ ARGS
###########################
model_weights= sys.argv[1]
filelist_img= "/home/riggi/Data/MLData/rg-dataset-yolo/datalists/crossval_RUN1.dat"
###filelist_img= "/home/riggi/Data/MLData/rgz-mask-dataset-yolo/dataset/images/val.txt"
###anndata_dir= "/home/riggi/Data/MLData/rgz-mask-dataset-yolo/dataset/labels/val"

draw= False
iou_thr= 0.6
merge_overlap_iou_thr_soft= 0.3
merge_overlap_iou_thr_hard= 0.8
score_thr= 0.5

###########################
##    MODEL LOAD
##########################
# Load a model
print("Loading model weights %s ..." % (model_weights))
model = YOLO(model_weights)

######################################
###   READ ANNOTATION DATA
######################################
fnames_img= read_filelist(filelist_img)
nimgs_max= -1

gtdata_list= []
counter= 0

for fname in fnames_img:
	fname= fname.strip()
	fname_base_noext= os.path.splitext(os.path.basename(fname))[0]
	print("Reading img %s ..." % (fname))

	# - Read image & get shape
	img= plt.imread(fname)
	h, w = img.shape[:2]

	# - Read annotation data
	fname_ann= fname.replace("/images/","/labels/").replace(".png",".txt")
	##fname_ann= os.path.join(anndata_dir, fname_base_noext + '.txt')
	print("Reading ann data file %s ..." % (fname_ann))

	d= {
		"img": fname,
		"ann": fname_ann,
		"bboxes": [],
		"labels": []
	}

	df= pd.read_csv(fname_ann, sep=" ", names=["class_id", "x_center", "y_center", "box_width", "box_height"])
	for ind in df.index:
		class_id= df['class_id'][ind]
		x_center= df['x_center'][ind] * w
		y_center= df['y_center'][ind] * h
		box_width= df['box_width'][ind] * w
		box_height= df['box_height'][ind] * h
		x1= x_center - 0.5*box_width
		y1= y_center - 0.5*box_height
		x2= x1 + box_width
		y2= y1 + box_height
		bbox= (x1, y1, x2, y2)
		label= model.names[int(class_id)]
		
		d["bboxes"].append(bbox)
		d["labels"].append(label)
	
	gtdata_list.append(d)	
		
	print("anndata")	
	print(d)
	
	counter+= 1
	if nimgs_max>0 and counter>=nimgs_max:
		break
	

		
###############################
##    PREDICT
###############################
draw= False
iou_thr= 0.6
merge_overlap_iou_thr_soft= 0.3
merge_overlap_iou_thr_hard= 0.8
score_thr= 0.25

#imgpath= "/home/riggi/Data/MLData/rgz-mask-dataset-yolo/dataset/images/val/galaxy0572_696.png"


# - Loop through gt data and make prediction
preddata_list= []
counter= 0

for i in range(len(fnames_img)):
	imgpath= fnames_img[i].strip()
	gtdata= gtdata_list[i]
	print("Making prediction for image %s ..." % (imgpath))
	
	results= predict(
		model, 
		imgpath,
		score_thr=score_thr,
		merge_overlap_iou_thr_soft=merge_overlap_iou_thr_soft,
		merge_overlap_iou_thr_hard=merge_overlap_iou_thr_hard, 
		gtdata=gtdata,
		draw=draw
	)

	bboxes_pred= results[0]
	labels_pred= results[1]
	scores_pred= results[2]

	print("bboxes_pred")
	print(bboxes_pred)
	print("labels_pred")
	print(labels_pred)
	print("scores_pred")
	print(scores_pred)
	
	preddata= {
		"img": imgpath,
		"bboxes": bboxes_pred,
		"labels": labels_pred,
		"scores": scores_pred
	}
	preddata_list.append(preddata)
	
	counter+= 1
	if nimgs_max>0 and counter>=nimgs_max:
		break

###############################
##    COMPUTE COMPLETENESS
###############################
compute_completeness(gtdata_list, preddata_list)

###############################
##    COMPUTE RELIABILITY
###############################
compute_reliability(gtdata_list, preddata_list)



