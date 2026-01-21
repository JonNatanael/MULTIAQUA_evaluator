# from PIL import Image
import numpy as np
import argparse, os
# from multiprocessing import Pool
from tqdm.auto import tqdm
import os.path as osp
import zipfile
import tempfile, shutil
import time

from multiaqua_eval import SemanticEvaluator
from multiaqua_eval.config import get_cfg
from multiaqua_eval.utils import TqdmPool

WORKERS=8

class MethodEvaluator():
	def __init__(self, cfg, evaluator):
		self.cfg = cfg
		self.evaluator = evaluator

	def evaluate_method(self, method):
		pred_dir = osp.join(self.cfg.PATHS.PREDICTIONS, method)
		output_dir = osp.join(self.cfg.PATHS.RESULTS, method)
		return self.evaluator.evaluate(pred_dir, output_dir, display_name=method)

def evaluate_zip():

	parser = argparse.ArgumentParser(description='MULTIAQUA evaluation script')

	parser.add_argument('results', help='results file or directory', type=str)
	parser.add_argument('--config', default='configs/multiaqua_semantic.yaml', type=str)

	args = parser.parse_args()

	zf = zipfile.ZipFile(args.results)

	with tempfile.TemporaryDirectory() as tempdir:
		zf.extractall(tempdir)
		cfg = get_cfg(args.config)
		os.makedirs(cfg.PATHS.RESULTS, exist_ok=True)

		evaluator = SemanticEvaluator(cfg)
		my_evaluator = MethodEvaluator(cfg, evaluator)
		# print(f'{cfg.PATHS.RESULTS=}')
		args.methods = [tempdir]
		# args.methods = ['results']

		# Calculate the start time
		start = time.time()

		results = my_evaluator.evaluate_method(args.methods[0])

		end = time.time()
		duration = end - start

		# Show the results : this can be altered however you like
		print(f"The evaluation finished in {duration:0.2f} seconds.")

		print(results)

		shutil.copy(f'{tempdir}/frames_val.csv', cfg.PATHS.RESULTS)
		try:
			shutil.copy(f'{tempdir}/frames_test.csv', cfg.PATHS.RESULTS)
		except:
			print("Evaluating on val subset, test summary not available.")
			pass
		shutil.copy(f'{tempdir}/summary.json', cfg.PATHS.RESULTS)

		print(f'Evaluation completed, detailed results are in: {cfg.PATHS.RESULTS}')

if __name__=='__main__':
	evaluate_zip()
