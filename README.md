# Box2Pix: An automatic box-level to pixel-level annotation transformation method for fire detection in remote sensing images
Box2Pix, an automatic method that integrates weak annotations such as text, point and box prompts with SAM, to convert object detection datasets into pixel-level semantic segmentation datasets.

[Ming Wang](https://github.com/OyamingO), Chenxiao Zhang*, Peng Yue*

[[`Paper`](https://github.com/OyamingO/Box2Pix)] [[`Project`](https://github.com/OyamingO/Box2Pix)] [[`Dataset`](https://pan.baidu.com/s/17geF8yl68Fibb7Ih44YgAQ?pwd=kv87)]  [[`BibTeX`](#Citing-Box2Pix)]


### 🔥: Installation

The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib
```

### Citing Box2Pix

If you use Box2Pix in your research, please use the following BibTeX entry.

```
@article{Box2Pix,
  title={Box2Pix: An automatic box-level to pixel-level annotation transformation method for fire detection in remote sensing images},
  author={Ming Wang, Peng Yue, Chenxiao Zhang},
  journal={GIScience & Remote Sensing},
  year={2024}
}

```
