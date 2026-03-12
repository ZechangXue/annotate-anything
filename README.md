**English** | [中文](README_zh-CN.md)

## DeepSurg Annotation Anything (Interactive annotation tool based on SAM2)

This project is an interactive medical image annotation tool built on **SAM2**.  
It supports multi-class masks, click/drag interactions, manual boundary refinement, polygon regions, and rich post-processing.  
With images and SAM2 checkpoints prepared, you can efficiently perform high-quality annotations on your local machine.

---

### Demo GIFs

All demo GIFs live in `docs/media/` and are rendered directly on the GitHub page.

#### 1. Basic interaction: click region + ENTER to confirm (video1 + video2)

![Basic click interaction 1](docs/media/video1.gif)

![Basic click interaction 2](docs/media/video2.gif)

#### 2. Manually add boundaries: Add boundary (video3)

When automatic segmentation is not ideal, or you want to insert an extra “cutting line”, use **Add boundary**:

![Add boundary](docs/media/video3.gif)

#### 3. Polygon region selection (video4)

Suitable for fuzzy or irregular boundaries, using multi-point polylines to precisely control the region:

![Polygon region](docs/media/video4.gif)

#### 4. Post-processing and gap filling (video5)

Shows post-processing and gap filling applied to all class masks after annotation:

![Post-processing](docs/media/video5.gif)

#### 5. Settings UI: language / colors / classes / model (video6)

Shows the Settings panel: switch UI language (EN/ZH), change class colors, add/remove classes, and switch SAM2 models:

![Settings](docs/media/video6.gif)

---

### Project structure

```text
annotation_anything/
├─ sam_interactive_segmentation.py   # Main entry script
├─ sam2/                             # SAM2 inference code
├─ configs/
│   └─ sam2/                         # SAM2 configuration files
├─ checkpoints/
│   └─ download_ckpts.sh             # SAM2 checkpoint 下载脚本（不含权重）
├─ images/                           # Input images (empty in repo)
├─ images_mask/                      # Intermediate results (optional, not committed)
├─ masks/                            # Output masks (empty in repo)
├─ docs/
│   └─ media/                        # Demo videos / GIFs
├─ requirements.txt                  # Python 依赖
├─ run_annotation.bat                # Windows 一键启动脚本
└─ README.md
```

> Note: `checkpoints/`, `images/`, `images_mask/`, and `masks/` are shipped without actual images or checkpoints; they act as path placeholders only.

---

### Environment setup

Python 3.9+ is recommended. It is best to install dependencies in a virtual environment:

```bash
cd annotation_anything

# 创建虚拟环境（可选）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

Key dependencies in `requirements.txt`:

- `torch`, `torchvision`: core deep learning framework  
- `numpy`, `Pillow`, `matplotlib`, `opencv-python`: numerical computation and image display  
- `scipy`: morphological / distance-transform based post-processing  
- `hydra-core`, `omegaconf`: used to build and configure SAM2 models  
- `segmentation-models-pytorch` (optional, for the Assist model)

---

### Model checkpoints

This repository **does not ship any model checkpoints**.  
Please download **SAM2.1** checkpoints into the `checkpoints/` directory.

#### 1. Download SAM2.1 checkpoints

From the project root:

```bash
cd checkpoints
bash download_ckpts.sh
```

The script downloads the following files from the official URLs into `checkpoints/`:

- `sam2.1_hiera_tiny.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_large.pt`

By default (configurable at the top of `sam_interactive_segmentation.py`):

```python
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_l.yaml"
```

You can switch to `small` or other variants depending on your GPU memory.

#### 2. (Optional) Assist model checkpoint

To enable the **Model assist** button, you need an extra segmentation model checkpoint, e.g.:

```python
ASSIST_MODEL_CHECKPOINT = "checkpoints/your_assist_model.ckpt"
```

Place your own checkpoint file under `checkpoints/` and set the correct path in the script or in the Settings UI.  
If you do not need Assist, you can ignore this feature.

---

### Data layout

By default, the tool expects the following directory structure (directories are created in the repo, but contain no images):

```text
annotation_anything/
├─ images/        # Input images to annotate (png/jpg, etc.)
├─ images_mask/   # Optional intermediate results
└─ masks/         # Output masks (single-channel label images)
```

Put all images to be annotated into the `images/` directory.

---

### How to run

#### Windows

```bash
# First time: install dependencies
pip install -r requirements.txt
```

After that you can simply double-click:

- `run_annotation.bat`: one-click launcher for the annotator.

Or run from the command line:

```bash
python sam_interactive_segmentation.py
```

#### macOS / Linux

```bash
pip install -r requirements.txt
python sam_interactive_segmentation.py
```

You can also write a small wrapper script `run_annotation.sh` that just runs `python sam_interactive_segmentation.py`.

---

### License & citation

- SAM2-related code and configs are taken from the official Meta repository and follow the original license.  
- The new interaction logic and annotation UI code in this project can be released under your chosen open-source license (e.g., MIT / Apache-2.0).

If you use this tool in academic work, please consider citing the official SAM2 paper and your own related publications.

