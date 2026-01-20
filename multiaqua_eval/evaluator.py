from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm.auto import tqdm
import pandas as pd
import json

import multiaqua_eval.context as ctx
import multiaqua_eval.metrics as M
# import multiaqua_eval.panoptic as PM


# def parse_annotations(json_data):
# 	# Prepare a image_name -> annotation dictionary
# 	id2img = {}
# 	for img in json_data['images']:
# 		id2img[img['id']] = osp.splitext(img['file_name'])[0]

# 	annotations = {}
# 	for ann in json_data['annotations']:
# 		img_name = id2img[ann['image_id']]
# 		annotations[img_name] = ann

# 	return annotations

class SemanticEvaluator():

	def __init__(self, cfg):
		self.cfg = cfg

		# Read image list
		with open(os.path.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.SUBSET_LIST), 'r') as file:
			self.image_list = [l.strip() for l in file]

		skip = 20
		self.image_list = self.image_list[::skip]

		self.iou_val = M.IoU(cfg)
		self.iou_test = M.IoU(cfg)

	def evaluate_image(self, mask_pred, seg_mask, pan_mask, pan_ann, image_name):
		"""Evaluates a single image

		Args:
			mask_pred (np.array): Predicted segmentation mask.
			seg_mask (np.array): GT segmentation mask.
			pan_mask (np.array): GT panoptic mask.
			pan_ann (list): GT panoptic annotations (COCO format).
			image_name (str): Name of the evaluated image (filename without extension).
		"""
		print(f'{image_name=}')
		H,W = seg_mask.shape
		# 1. Resize predicted mask
		mask_pred = np.array(Image.fromarray(mask_pred).resize((W, H), Image.NEAREST))

		# 2. Evaluate IoU
		if 'lj4' in image_name:
			frame_summary, overall_summary = self.iou_test.compute(mask_pred, seg_mask, pan_mask, pan_ann, image_name)
			# frame_summary_mar, overall_summary_mar = self.maritime_metrics_test.compute(mask_pred, seg_mask, pan_mask, pan_ann, image_name)

		else:
			frame_summary, overall_summary = self.iou_val.compute(mask_pred, seg_mask, pan_mask, pan_ann, image_name)
			# frame_summary_mar, overall_summary_mar = self.maritime_metrics_val.compute(mask_pred, seg_mask, pan_mask, pan_ann, image_name)

		return frame_summary, overall_summary

	def evaluate(self, preds_dir, output_dir, display_name=None):
		sem_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.SEMANTIC_MASK_SUBDIR)
		pan_dir = os.path.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.PANOPTIC_MASK_SUBDIR)

		frame_results_val = []
		frame_results_test = []

		with tqdm(desc=display_name, total=len(self.image_list), position=ctx.PID, leave=False) as pbar:
			for img_name in self.image_list:
				try:
					mask_pred_c = np.array(Image.open(os.path.join(preds_dir, '%s.png' % img_name)).convert('RGB'))
				except:
					raise FileNotFoundError(img_name)
					# continue
				# mask_pred = np.array(Image.open(os.path.join(preds_dir, '%s.png' % img_name)))[...,0]
				# print(f'{np.unique(mask_pred_c, axis=2)=}')

				# Convert color mask to class ID
				H,W,_ = mask_pred_c.shape
				# H,W = mask_pred.shape
				mask_pred = np.full((H,W), self.cfg.SEGMENTATION.IGNORE_ID, np.uint8)
				for cls_i, cls_c in zip(self.cfg.SEGMENTATION.IDS, self.cfg.SEGMENTATION.COLORS):
					mask_cur = (mask_pred_c == np.array(cls_c)).all(2)
					mask_pred[mask_cur] = cls_i+1

				mask_sem = np.array(Image.open(os.path.join(sem_dir, '%s.png' % img_name)))
				# mask_pan = np.array(Image.open(os.path.join(pan_dir, '%s.png' % img_name)))
				mask_pan = mask_sem
				# ann_pan = self.annotations[img_name]

				frame_summary, overall_summary = self.evaluate_image(mask_pred, mask_sem, mask_pan, None, img_name)
				frame_summary['image'] = img_name
				if 'lj4' in img_name:
					frame_results_test.append(frame_summary)
				else:					
					frame_results_val.append(frame_summary)

				log_dict = {m:overall_summary[m] for m in self.cfg.PROGRESS.METRICS}

				# print(log_dict)

				pbar.set_postfix(log_dict)
				pbar.update()

		frame_results_val_df = pd.DataFrame(frame_results_val).set_index('image')
		frame_results_test_df = pd.DataFrame(frame_results_test).set_index('image')
		overall_summary_val = self.iou_val.summary()
		overall_summary_test = self.iou_test.summary()
		# print(f'{overall_summary_val=}')
		# print(f'{overall_summary_test=}')

		overall_summary = {
			'val_mIoU': overall_summary_val['mIoU'],
			'val_obstacle': overall_summary_val['IoU_dynamic_obstacle'],
			'test_mIoU': overall_summary_test['mIoU'],
			'test_obstacle': overall_summary_test['IoU_dynamic_obstacle'],
			'M': (overall_summary_val['mIoU']+overall_summary_test['mIoU'])/2
		}

		# print(f'{overall_summary=}')

		if not osp.exists(output_dir):
			os.makedirs(output_dir)

		frame_results_val_df.to_csv(osp.join(output_dir, 'frames_val.csv'))
		frame_results_test_df.to_csv(osp.join(output_dir, 'frames_test.csv'))
		with open(osp.join(output_dir, 'summary.json'), 'w') as file:
			json.dump(overall_summary, file, indent=2)

		return overall_summary