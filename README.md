# Det2Seg: An automatic pixel-level Labeling method based on multi-scale prompts for fire detection in remote sensing images	
Det2Seg, an automatic method that integrates weak annotations such as text, point and box prompts with SAM, to convert object detection datasets into pixel-level semantic segmentation datasets.

[Ming Wang](https://github.com/OyamingO), Chenxiao Zhang*, Peng Yue*

[[`Paper`](https://github.com/OyamingO/Det2Seg)] [[`Project`](https://github.com/OyamingO/Det2Seg)] [[`Dataset`](https://github.com/OyamingO/Det2Seg)]  [[`BibTeX`](#Citing-Det2Seg)]


### 🔥: Installation

The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib
```

### 🚀: Citing Det2Seg

If you use Det2Seg in your research, please use the following BibTeX entry.

```
@article{Det2Seg,
  title={Det2Seg: An automatic pixel-level Labeling method based on multi-scale prompts for fire detection in remote sensing images},
  author={Ming Wang, Chenxiao Zhang, Peng Yue},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  year={2024}
}

```
