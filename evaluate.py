from PIL import Image
import numpy as np
import argparse, os
from multiprocessing import Pool
from tqdm.auto import tqdm
import os.path as osp
import zipfile
import tempfile, glob, shutil

from lars_eval import SemanticEvaluator
from lars_eval.config import get_cfg
from lars_eval.utils import TqdmPool

WORKERS=8

class MethodEvaluator():
	def __init__(self, cfg, evaluator):
		self.cfg = cfg
		self.evaluator = evaluator

	def evaluate_method(self, method):
		pred_dir = osp.join(self.cfg.PATHS.PREDICTIONS, method)
		output_dir = osp.join(self.cfg.PATHS.RESULTS, method)
		# print(f'{output_dir=}')
		return self.evaluator.evaluate(pred_dir, output_dir, display_name=method)


def main():
	parser = argparse.ArgumentParser(description='LaRS evaluation script')

	parser.add_argument('config', help='Configuration file', type=str)
	# parser.add_argument('methods', nargs='+', help='Method(s) to evaluate. Prediction dir should contain a directory with the same name, containing the predicted segmentation masks',type=str)

	parser.add_argument('--workers', default=WORKERS, type=int)



	args = parser.parse_args()

	# args.methods = ['/media/jon/disk2/code/DELIVER/development_output/MULTIAQUA_MiT_itl/20250805145916_loss=CE_lr=6e-05_DROP_MODALITIES=True_SHARED_BACKBONE=True_FUSION_MODE=naive_DECODER_HEAD=Mask2Former_MULTIHEAD=False/predictions/test/320x576/itl/']
	args.methods = ['/media/jon/disk2/code/DELIVER/development_output/MULTIAQUA_MiT_itl/20250826170057_loss=CE_lr=6e-04_DROP_MODALITIES=True_SHARED_BACKBONE=True_FUSION_MODE=naive_DECODER_HEAD=UPerHead_MULTIHEAD=False/predictions/test/1242x2208/itl/']
	# args.methods = ['/media/jon/disk2/code/DELIVER/development_output/MULTIAQUA_MiT_itl/20250805145916_loss=CE_lr=6e-05_DROP_MODALITIES=True_SHARED_BACKBONE=True_FUSION_MODE=naive_DECODER_HEAD=Mask2Former_MULTIHEAD=False/predictions/val/320x576/itl/']


	cfg = get_cfg(args.config)

	if cfg.MODE == 'semantic':
		evaluator = SemanticEvaluator(cfg)
	else:
		raise ValueError('Unknown mode: %s' % cfg.MODE)

	my_evaluator = MethodEvaluator(cfg, evaluator)

	if len(args.methods) > 1:
		with TqdmPool(WORKERS) as pool:
			list(tqdm(pool.imap_unordered(my_evaluator.evaluate_method, args.methods), total=len(args.methods)))
	else:
		results = my_evaluator.evaluate_method(args.methods[0])
		print(results)

def evaluate_zip(zip_path):

	# archive = zipfile.ZipFile(zip_path, 'r')
	# imgdata = archive.read('lj4_1_067050.png')
	# img = cv2.imread(imgdata)
	# print(archive)

	parser = argparse.ArgumentParser(description='LaRS evaluation script')

	parser.add_argument('config', help='Configuration file', type=str)
	# parser.add_argument('methods', nargs='+', help='Method(s) to evaluate. Prediction dir should contain a directory with the same name, containing the predicted segmentation masks',type=str)
	# parser.add_argument('--workers', default=WORKERS, type=int)

	args = parser.parse_args()

	zf = zipfile.ZipFile(zip_path)


	with tempfile.TemporaryDirectory() as tempdir:
		zf.extractall(tempdir)

		print(tempdir)
		cfg = get_cfg(args.config)
		os.makedirs(cfg.PATHS.RESULTS, exist_ok=True)

		evaluator = SemanticEvaluator(cfg)
		my_evaluator = MethodEvaluator(cfg, evaluator)
		print(f'{cfg.PATHS.RESULTS=}')
		args.methods = [tempdir]
		# args.methods = ['results']
		results = my_evaluator.evaluate_method(args.methods[0])
		print(results)

		# print(glob.glob(tempdir+'/*'))

		shutil.copy(f'{tempdir}/frames_val.csv', cfg.PATHS.RESULTS)
		shutil.copy(f'{tempdir}/frames_test.csv', cfg.PATHS.RESULTS)
		shutil.copy(f'{tempdir}/summary.json', cfg.PATHS.RESULTS)

if __name__=='__main__':
	# main()
	# zip_path = 'jon_predictions.zip'
	zip_path = 'predictions_testing.zip'
	evaluate_zip(zip_path)
