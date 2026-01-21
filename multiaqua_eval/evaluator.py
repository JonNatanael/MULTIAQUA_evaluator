from PIL import Image
import numpy as np
import os
import os.path as osp
from tqdm.auto import tqdm
import pandas as pd
import json

import multiaqua_eval.context as ctx
import multiaqua_eval.metrics as M

class SemanticEvaluator():

	def __init__(self, cfg):
		self.cfg = cfg

		# Read image list
		with open(osp.join(cfg.PATHS.DATASET_ROOT, cfg.DATASET.SUBSET_LIST), 'r') as file:
			self.image_list = [l.strip() for l in file]

		# skip = 20
		# self.image_list = self.image_list[::skip]

		self.iou_val = M.IoU(cfg)
		self.iou_test = M.IoU(cfg)

	def evaluate_image(self, mask_pred, seg_mask, image_name):
		"""Evaluates a single image

		Args:
			mask_pred (np.array): Predicted segmentation mask.
			seg_mask (np.array): GT segmentation mask.
			pan_mask (np.array): GT panoptic mask.
			pan_ann (list): GT panoptic annotations (COCO format).
			image_name (str): Name of the evaluated image (filename without extension).
		"""
		H, W = seg_mask.shape
		H_pred, W_pred = mask_pred.shape		

		# 1. assert size matching
		assert H==H_pred and W==W_pred, "The prediction size must match the GT size"

		# 2. Evaluate IoU
		if 'lj4' in image_name:
			frame_summary, overall_summary = self.iou_test.compute(mask_pred, seg_mask, image_name)
		else:
			frame_summary, overall_summary = self.iou_val.compute(mask_pred, seg_mask, image_name)

		return frame_summary, overall_summary

	def evaluate(self, preds_dir, output_dir, display_name=None):
		sem_dir = osp.join(self.cfg.PATHS.DATASET_ROOT, self.cfg.DATASET.SEMANTIC_MASK_SUBDIR)

		frame_results_val = []
		frame_results_test = []

		# check if color or raw
		base_img = Image.open(osp.join(preds_dir, '%s.png' % self.image_list[0]))

		use_raw_format = base_img.mode=='L'

		with tqdm(desc=display_name, total=len(self.image_list), position=ctx.PID, leave=False) as pbar:
			for img_name in self.image_list:

				try:
					image = Image.open(osp.join(preds_dir, '%s.png' % img_name))
				except:
					raise FileNotFoundError(img_name)
				
				if use_raw_format:
					mask_pred = np.array(image)					
				else:					
					# Convert color mask to class ID
					mask_pred_c = np.array(image.convert('RGB'))

					H,W,_ = mask_pred_c.shape
					mask_pred = np.full((H,W), self.cfg.SEGMENTATION.IGNORE_ID, np.uint8)
					for cls_i, cls_c in zip(self.cfg.SEGMENTATION.IDS, self.cfg.SEGMENTATION.COLORS):
						mask_cur = (mask_pred_c == np.array(cls_c)).all(2)
						mask_pred[mask_cur] = cls_i

				mask_sem = np.array(Image.open(osp.join(sem_dir, '%s.png' % img_name)))

				frame_summary, overall_summary = self.evaluate_image(mask_pred, mask_sem, img_name)
				frame_summary['image'] = img_name
				if 'lj4' in img_name:
					frame_results_test.append(frame_summary)
				else:					
					frame_results_val.append(frame_summary)

				log_dict = {m:overall_summary[m] for m in self.cfg.PROGRESS.METRICS}

				pbar.set_postfix(log_dict)
				pbar.update()

		frame_results_val_df = pd.DataFrame(frame_results_val).set_index('image')
		if frame_results_test:
			frame_results_test_df = pd.DataFrame(frame_results_test).set_index('image')
		overall_summary_val = self.iou_val.summary()
		if frame_results_test:
			overall_summary_test = self.iou_test.summary()
		else:
			overall_summary_test = {'mIoU': 0.0, 'IoU_dynamic_obstacle': 0.0}

		overall_summary = {
			'val_mIoU': overall_summary_val['mIoU'],
			'val_obstacle': overall_summary_val['IoU_dynamic_obstacle'],
			'test_mIoU': overall_summary_test['mIoU'],
			'test_obstacle': overall_summary_test['IoU_dynamic_obstacle'],
			'M': (overall_summary_val['mIoU']+overall_summary_test['mIoU'])/2
		}

		if not osp.exists(output_dir):
			os.makedirs(output_dir)

		frame_results_val_df.to_csv(osp.join(output_dir, 'frames_val.csv'))
		if frame_results_test:
			frame_results_test_df.to_csv(osp.join(output_dir, 'frames_test.csv'))

		with open(osp.join(output_dir, 'summary.json'), 'w') as file:
			json.dump(overall_summary, file, indent=2)

		return overall_summary