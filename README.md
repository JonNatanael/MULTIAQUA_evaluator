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
2. The evaluator expects a zip file containing prediction images.
Configure the path to the dataset in the config file (*e.g.* [multiaqua_semantic.yaml.yaml](configs/multiaqua_semantic.yaml)).

## Usage

1. Place the predictions of your methods into `<prediction_root_dir>/<method_name>`
    The method dir contains PNG files with predictions for all test images:
    - **Semantic segmentation**: The PNG file contains the predicted segmentation in RGB format, following the color coding of classes specified in the configuration file (*e.g.* [lars_val_semantic.yaml](configs/v1.0.0/lars_val_semantic.yaml)). By default this is:
        - sky: `[90,  75, 164]`
        - water: `[41, 167, 224]`
        - obstacle: `[247, 195,  37]`
        - Alternatively you may use the [lars_val_semantic_lbl.yaml](configs/v1.0.0/lars_val_semantic_lbl.yaml) config to evaluate predictions encoded as class ids (0 = obstacles, 1 = water, 2 = sky). Note, however, that the online evaluator expects predictions in the **color-coded format**.
    - **Panoptic segmentation**: The PNG file contains RGB coded class and instance predictions. The format follows LaRS GT masks: *class id* is stored in the **R** component, while *instance ids* are stored in the **G** and **B** components. 
2. Run evaluation:
    ```bash
    $ python evaluate.py path/to/config.yaml <method_name>
    ```

> [!NOTE]
> Result files with various statistics will be placed in the configured directory (`results/v1.0.0/<track>/<method>` by default).

## Evaluation server

You can evaluate your methods on the MULTIAQUA **test** set through our online [evaluation server](https://macvi.org/). You need to create an account to submit your results.

The server runs the same code as this evaluator and expects the same prediction format. The predictions of a single method need to be stored into a .zip archive and submitted to the server. [More information >](https://macvi.org/dataset#MULTIAQUA)


## Result files

### Semantic segmentation

Results for semantic segmentation methods inlcude the following files:

- `summary.csv`: Overall results (IoU, water-edge accuracy, detection F1)
- `frames_val.csv`: Per frame metrics for the validation subset (number of TP, FP and FN, IoU, ...)
- `frames_test.csv`: Per frame metrics for the test subset (number of TP, FP and FN, IoU, ...) *Evaluation server only*

## <a name="cite"></a>Citation

If you use LaRS, please cite our paper.

```bibtex
@article{muhovivc2025multiaqua,
  title={MULTIAQUA: A multimodal maritime dataset and robust training strategies for multimodal semantic segmentation},
  author={Muhovi{\v{c}}, Jon and Per{\v{s}}, Janez},
  journal={arXiv preprint arXiv:2512.17450},
  year={2025}
}
```
