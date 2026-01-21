# MULTIAQUA Evaluator

[![arxiv](https://img.shields.io/badge/paper-52b69a?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.17450)
[![dataset](https://img.shields.io/badge/dataset-34a0a4?style=for-the-badge&logo=DocuSign&logoColor=white)](https://lmi.fe.uni-lj.si/en/MULTIAQUA/)
[![server](https://img.shields.io/badge/evaluation_server-168aad?style=for-the-badge&logo=windowsterminal&logoColor=white)](https://macvi.org)
[![cite](https://img.shields.io/badge/bibtex-1a759f?style=for-the-badge&logo=overleaf&logoColor=white)](#cite)

<!-- <p align="center">
    <img src="lars_ex.jpg" alt="LaRS Examples">
    Examples of scenes in the LaRS benchmark.
</p> -->

This is the evaluator code for the paper "MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation". It can be used to evaluate multimodal semantic segmentation predictions with the MULTIAQUA ground-truth annotations. 

Currently only the GT of the *training* and *validation* sets is publicly available. For evaluation on the MULTIAQUA test set, please submit your submissions through [our evaluation server](https://macvi.org).

## Setup

1. Install requirements into your python environment
    ```bash
    pip install -r requirements.txt
    ```
2. Configure the path to the MULTIAQUA dataset in the config file (*e.g.* [multiaqua_semantic.yaml.yaml](configs/multiaqua_semantic.yaml)).



## Usage

The evaluator consists of two main scripts: `evaluate.py` and `evaluate_zip.py`. They perform the same evaluation, but `evaluate.py` can be run on multiple prediction directories simultaneously. The script `evaluate_zip.py` mirrors the behavior on the evaluation server and expects one zip file with predictions for a single method.

1. Place the predictions of your methods into `<prediction_root_dir>/<method_name>`
    The method directory should contain PNG files with predictions for all val/test images:

    Each PNG file should match the size of the MULTIAQUA annotation files and contain the predicted semantic segmentations. The format of the predictions can be raw (exactly matching the GT annotations) or in RGB format, following the color coding of classes specified in the configuration file (*e.g.* [multiaqua_semantic.yaml](configs/multiaqua_semantic.yaml)). By default this is:
        - static obstacle `[0, 255, 0]`
        - dynamic obstacle `[255, 0, 0]`
        - water `[0, 0, 255]`
        - sky: `[148, 0, 211]`

    The color-mapped predictions are easier to interpret by eye, but the prediction scripts work much faster for raw predictions.
    
2. Run evaluation:
    ```bash
    $ python evaluate.py <method_name> --config path/to/config.yaml
    $ python evaluate_zip.py <predictions_zip>.zip --config path/to/config.yaml
    ```

> [!NOTE]
> Result files with various statistics will be placed in the configured directory (`results/MULTIAQUA_semantic` by default).

## Evaluation server

You can evaluate your methods on the MULTIAQUA **test** set through our online [evaluation server](https://macvi.org/). You need to create an account to submit your results.

The server runs the same code as this evaluator and expects the same prediction format. The predictions of a single method need to be stored into a .zip archive and submitted to the server. [More information >](https://macvi.org/dataset#MULTIAQUA)


## Result files

### Semantic segmentation

Results for semantic segmentation methods inlcude the following files:

- `summary.csv`: Overall results (mIoU, dynamic obstacle IoU, M)
- `frames_val.csv`: Per frame metrics for the validation subset (mIoU, per-class IoU)
- `frames_test.csv`: Per frame metrics for the test subset (mIoU, per-class IoU) **Evaluation server only**

## <a name="cite"></a>Citation

If you use MULTIAQUA, please cite our paper.

```bibtex
@article{muhovivc2025multiaqua,
  title={MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation},
  author={Muhovi{\v{c}}, Jon and Per{\v{s}}, Janez},
  journal={arXiv preprint arXiv:2512.17450},
  year={2025}
}
```
