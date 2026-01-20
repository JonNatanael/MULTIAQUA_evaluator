

import numpy as np
import cv2
import json
import os

from multiaqua_eval.panopticapi import rgb2id

from matplotlib import pyplot as plt

def _get_diagonal(bbox):
	return np.sqrt(bbox[2]**2 + bbox[3]**2)

class Metric():
	def compute(self, mask_pred, mask_gt, **kwargs):
		pass

	def summary(self):
		pass

	def reset(self):
		pass

	def save_extras(self, path, **kwargs):
		pass

class IoU(Metric):
	def __init__(self, cfg):
		self.classes = cfg.SEGMENTATION.IDS
		self.class_names = cfg.SEGMENTATION.NAMES
		self.ignore_idx = cfg.SEGMENTATION.IGNORE_ID

		print(f'{self.ignore_idx=}')

		self.reset()

	def reset(self):
		# Metric counters
		self._total_union = {cls_i: 0 for cls_i in self.classes}
		self._total_intersection = {cls_i: 0 for cls_i in self.classes}

	def compute(self, mask_pred, gt_sem, gt_pan, ann_pan, image_name):
		frame_summary = {}
		for i,cls_i in enumerate(self.classes):
			# print(f'{i=}')
			# print(f'{cls_i=}')
			cls_pred = (mask_pred == cls_i+1) & (gt_sem != self.ignore_idx)
			cls_gt = gt_sem == cls_i+1

			# plt.clf()
			# plt.subplot(2,2,1)
			# plt.imshow(mask_pred)
			# plt.title('mask_pred')
			# plt.subplot(2,2,2)
			# plt.imshow(gt_sem)
			# plt.title('gt_sem')
			# plt.subplot(2,2,3)
			# plt.imshow(cls_pred)
			# plt.title('cls_pred')
			# plt.subplot(2,2,4)
			# plt.imshow(cls_gt)
			# plt.title('cls_gt')

			# plt.show()

			intersection = np.bitwise_and(cls_pred, cls_gt).sum()
			union = np.bitwise_or(cls_pred, cls_gt).sum()

			self._total_intersection[cls_i] += intersection
			self._total_union[cls_i] += union

			# Store current frame IoU
			cls_name = self.class_names[i] if self.class_names is not None else '%d' % cls_i
			frame_summary['IoU_%s' % cls_name] = 100. * intersection / union if union != 0 else 100.

		frame_summary['mIoU'] = sum(frame_summary.values()) / len(frame_summary)

		# Return current frame summary and overall summary
		return frame_summary, self.summary()

	def summary(self):
		results = {}
		for i, cls_i in enumerate(self.classes):
			cls_iou = 100. * self._total_intersection[cls_i] / self._total_union[cls_i]
			cls_name = self.class_names[i] if self.class_names is not None else '%d' % cls_i
			results['IoU_%s' % cls_name] = cls_iou

		results['mIoU'] = sum(results.values()) / len(results)
		return results

def dilate_mask(mask, ksize=3, it=1):
	kernel = np.ones((ksize,ksize), np.uint8)
	out = cv2.dilate(mask, kernel, iterations=it)
	return out

def erode_mask(mask, ksize=3, it=1):
	kernel = np.ones((ksize,ksize), np.uint8)
	out = cv2.erode(mask, kernel, iterations=it)
	return out