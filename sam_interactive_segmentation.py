#!/usr/bin/env python3
"""
Interactive medical image segmentation using SAM2.

Iterates over images in an 'images' folder, lets the user annotate predefined
classes with point prompts, then saves multi-class masks to masks/<image_name>.png.
Uses your existing SAM2 checkpoints (sam2.1_hiera_large.pt / sam2.1_hiera_small.pt).
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg for responsive interactive display
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
try:
    # 优先使用支持中文的字体，避免主界面中文乱码；若不存在则保持默认字体
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
except Exception:
    pass
from PIL import Image

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except Exception:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None


def _resource_path(relative_path: str) -> str:
    """
    Resolve a resource path (configs, checkpoints, etc.) that works both
    in normal Python execution and inside a PyInstaller bundle.
    """
    # When frozen by PyInstaller, use _MEIPASS / executable folder first
    if getattr(sys, "frozen", False):
        # 1) Try inside the internal bundle directory (_MEIPASS)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cand = os.path.join(meipass, relative_path)
            if os.path.exists(cand):
                return cand
        # 2) Try next to the executable (dist/DeepSurgAnnotator/...)
        exe_dir = os.path.dirname(sys.executable)
        cand = os.path.join(exe_dir, relative_path)
        if os.path.exists(cand):
            return cand

    # Non-frozen (normal Python): resolve relative to this file / its parent
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, relative_path)
    if os.path.exists(cand):
        return cand
    parent = os.path.dirname(here)
    cand = os.path.join(parent, relative_path)
    if os.path.exists(cand):
        return cand

    # Fallback: just treat as relative to current working directory
    return os.path.abspath(relative_path)


# -----------------------------------------------------------------------------
# SAM2 config: use one of your two checkpoints
# -----------------------------------------------------------------------------
# Large: better accuracy, ~6-8GB VRAM
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Small: faster, ~2-3GB VRAM（现在可在设置里修改，无需改代码）
# SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
# SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Run from tracking dir: python sam_interactive_segmentation.py

# Mask post-process: remove small sprinkles / fill holes (low-res pixel area)
SAM2_MAX_SPRINKLE_AREA = 150
SAM2_MAX_HOLE_AREA = 150
# After generating mask: remove components smaller than this (needs scipy)
MIN_MASK_COMPONENT_PIXELS = 80
# Smooth jagged edges: morphological closing with disk of this radius (0 = off)
SMOOTH_EDGE_RADIUS = 2

# Before save: assign all unlabeled pixels to nearest class (no gaps left)
FILL_GAPS_ENABLED = True
FILL_GAPS_MAX_DISTANCE = 25  # unused; kept for compatibility
# Gap-fill never assigns to these classes (e.g. instruments); use class index 0-based
FILL_GAPS_EXCLUDE_CLASSES = [
    5,   # Grasper
    9,   # L-hook Electrocautery
    13,  # Clipper
    14,  # Scissors
    15,  # Irrigator
    16,  # Bipolar
    17,  # SpecimenBag
]

# Drag-to-draw boundary: sample a point every N pixels along the stroke
STROKE_SAMPLE_INTERVAL_PIXELS = 8
# Min stroke length (pixels) to count as draw, else single click
STROKE_MIN_LENGTH_PIXELS = 6
# When user draws a long stroke that roughly closes back to the start within this distance,
# treat it as a freehand region and use the enclosed area directly as a mask (no SAM).
FREEHAND_CLOSE_DISTANCE_PIXELS = 25

# Manually drawn extra boundary lines: line thickness in pixels
BOUNDARY_LINE_WIDTH_PIXELS = 3

# Per-class final post-processing config (applied once at Finish image).
# Keys use 0-based class index; values are (min_component_pixels, smooth_radius).
# Unlisted classes fall back to (MIN_MASK_COMPONENT_PIXELS, SMOOTH_EDGE_RADIUS).
CLASS_POSTPROCESS_CONFIG = {
    # Large smooth organs: stronger smoothing
    1: (200, 3),   # Abdominal Wall
    2: (200, 3),   # Liver
    3: (200, 3),   # Gastrointestinal
    4: (200, 3),   # Fat
    6: (200, 3),   # Connective Tissue
    7: (150, 3),   # Blood (can be a bit noisier)
    10: (150, 3),  # Gallbladder
    11: (150, 3),  # Hepatic Vein (mild smoothing)
    12: (150, 3),  # Liver Ligament
    # Instruments / background: keep edges sharper, mainly remove tiny speckles
    0: (MIN_MASK_COMPONENT_PIXELS, 1),   # Black Background
    5: (MIN_MASK_COMPONENT_PIXELS, 1),   # Grasper
    9: (MIN_MASK_COMPONENT_PIXELS, 1),   # L-hook Electrocautery
    13: (MIN_MASK_COMPONENT_PIXELS, 1),  # Clipper
    14: (MIN_MASK_COMPONENT_PIXELS, 1),  # Scissors
    15: (MIN_MASK_COMPONENT_PIXELS, 1),  # Irrigator
    16: (MIN_MASK_COMPONENT_PIXELS, 1),  # Bipolar
    17: (MIN_MASK_COMPONENT_PIXELS, 1),  # SpecimenBag
}

# Optional model-assisted labeling (for Fat, Blood, etc.): trained segmentation model + box
ASSIST_MODEL_ENABLED = True
ASSIST_MODEL_CHECKPOINT = "checkpoints/reproduce_best-epoch=76-valid_dataset_iou=0.9579.ckpt"

# Folders
IMAGES_DIR = "images"
MASKS_DIR = "masks"
IMAGES_MASK_DIR = "images_mask"

# Predefined medical classes:
#   - Index 0..12: also used by the segmentation assist model (13 classes total)
#   - Index >=13: extra instruments that are NOT in the assist model
CLASSES = [
    "Black Background",       # 0
    "Abdominal Wall",         # 1
    "Liver",                  # 2
    "Gastrointestinal",       # 3
    "Fat",                    # 4
    "Grasper",                # 5
    "Connective Tissue",      # 6
    "Blood",                  # 7
    "Cystic Duct",            # 8
    "L-hook Electrocautery",  # 9
    "Gallbladder",            # 10
    "Hepatic Vein",           # 11
    "Liver Ligament",         # 12
    "Clipper",                # 13
    "Scissors",               # 14
    "Irrigator",              # 15
    "Bipolar",                # 16
    "SpecimenBag",            # 17
]
# Annotation order: bottom layer first, then overlay (fat/connective/blood),
# instruments (including new ones) last. Later overwrites earlier.
ANNOTATION_ORDER = [0, 2, 10, 11, 12, 3, 8, 1, 4, 6, 7, 5, 9, 13, 14, 15, 16, 17]

# Preview overlay when asking Accept mask? (visible on all backgrounds)
PREVIEW_OVERLAY_COLOR = (0.0, 0.9, 1.0, 0.55)
PREVIEW_EDGE_COLOR = (1.0, 1.0, 0.0, 1.0)

# Per-class colors for saved images_mask overlay
OVERLAY_COLORS = [
    (0.2, 0.2, 0.2, 0.6),   # Black Background
    (1.0, 0.4, 0.4, 0.6),   # Abdominal Wall
    (0.4, 0.8, 0.4, 0.6),   # Liver
    (0.4, 0.4, 1.0, 0.6),   # Gastrointestinal
    (1.0, 0.9, 0.4, 0.6),   # Fat
    (0.9, 0.5, 0.0, 0.6),   # Grasper
    (0.6, 0.6, 0.8, 0.6),   # Connective Tissue
    (1.0, 0.2, 0.2, 0.6),   # Blood
    (0.4, 0.8, 0.8, 0.6),   # Cystic Duct
    (0.8, 0.4, 1.0, 0.6),   # L-hook Electrocautery
    (0.2, 0.6, 1.0, 0.6),   # Gallbladder
    (0.0, 0.8, 0.6, 0.6),   # Hepatic Vein
    (0.8, 0.6, 0.2, 0.6),   # Liver Ligament
    (1.0, 0.7, 0.2, 0.6),   # Clipper
    (0.9, 0.3, 0.7, 0.6),   # Scissors
    (0.3, 0.7, 1.0, 0.6),   # Irrigator
    (0.7, 1.0, 0.3, 0.6),   # Bipolar
    (0.6, 0.6, 0.1, 0.6),   # SpecimenBag
]

SETTINGS_FILE = "deep_surg_config.json"

# UI language: "en" or "zh" (affects Settings dialog labels only; main figure uses ASCII-safe text)
UI_LANGUAGE = "en"

# 标记设置是否在运行期间被修改，用于触发模型重载等
_settings_dirty = False


def _refresh_main_ui_language(fig):
    """
    根据当前 UI_LANGUAGE，刷新主标注窗口上已有控件的文字（Classes 标题、底部按钮等），
    这样在 Settings 中切换语言后无需重启就能立刻看到变化。
    """
    try:
        if hasattr(fig, "_ax_toolbar") and fig._ax_toolbar is not None:
            fig._ax_toolbar.set_title(_ui_text("toolbar_classes"), fontsize=10)
        # 更新当前图像上方提示行
        if hasattr(fig, "_ax_display") and fig._ax_display is not None:
            title = fig._ax_display.get_title() or ""
            if "ENTER" in title or "点击或拖动轮廓" in title:
                fig._ax_display.set_title(_ui_text("click_or_drag"))

        # 通过保存的按钮引用刷新底部按钮及设置按钮、模型辅助按钮
        if hasattr(fig, "_preview_btn") and fig._preview_btn is not None:
            fig._preview_btn.label.set_text(
                _ui_text("preview_hide")
                if getattr(fig, "_preview_showing", False)
                else _ui_text("preview_show")
            )
        if hasattr(fig, "_finish_btn") and fig._finish_btn is not None:
            fig._finish_btn.label.set_text(_ui_text("finish_image"))
        if hasattr(fig, "_settings_btn") and fig._settings_btn is not None:
            fig._settings_btn.label.set_text(_ui_text("settings_button"))
        if hasattr(fig, "_model_assist_btn") and fig._model_assist_btn is not None:
            fig._model_assist_btn.label.set_text(_ui_text("model_assist_button"))
        if hasattr(fig, "_boundary_btn") and fig._boundary_btn is not None:
            fig._boundary_btn.label.set_text(_ui_text("boundary_button"))
        if hasattr(fig, "_polygon_btn") and fig._polygon_btn is not None:
            fig._polygon_btn.label.set_text(_ui_text("polygon_button"))

        # 初始状态栏提示改成当前语言
        _update_status_line(fig, "", _ui_text("main_initial"), "")
        fig.canvas.draw_idle()
    except Exception:
        pass


def _ui_text(key: str) -> str:
    """
    Simple runtime i18n helper for main annotation UI.
    Only affects界面文案，不改变 classes / 模型等真实含义。
    """
    if UI_LANGUAGE == "zh":
        zh = {
            "toolbar_classes": "类别",
            "main_initial": "请在此窗口操作：ENTER=确认，ESC=取消",
            "click_or_drag": "点击或拖动轮廓 —— ENTER=完成，ESC=跳过 —— 或使用 Model assist",
            "status_hint": "使用下方工具栏按钮",
            "prompt_click": "点击或拖动轮廓",
            "status_current_prefix": "当前：",
            "preview_show": "显示预览",
            "preview_hide": "隐藏预览",
            "finish_image": "完成当前图像",
            "model_assist_button": "模型辅助 (框选)",
            "settings_button": "设置",
            "boundary_button": "新增边界",
            "polygon_button": "多点连线",
        }
        return zh.get(key, key)
    # 默认英文
    en = {
        "toolbar_classes": "Classes",
        "main_initial": "Operate in this window. ENTER=Yes ESC=No",
        "click_or_drag": "Click or drag boundary — ENTER=done, ESC=skip — or use Model assist",
        "status_hint": "Use toolbar buttons below",
        "prompt_click": "Click or drag boundary",
        "status_current_prefix": "Current: ",
        "preview_show": "Show preview",
        "preview_hide": "Hide preview",
        "finish_image": "Finish image",
        "model_assist_button": "Model assist (box)",
        "settings_button": "Settings",
        "boundary_button": "Add boundary",
        "polygon_button": "Polygon region",
    }
    return en.get(key, key)

# Global flag: set when user closes the figure (skip to next image)
_window_closed = False


def _load_settings():
    """
    从 JSON 配置文件加载可编辑设置（classes 名称与颜色、SAM / assist checkpoint、标注顺序），
    如果文件不存在或字段缺失，则用当前脚本中的默认值，并写出一个默认配置文件。
    """
    global SAM2_CHECKPOINT, SAM2_CONFIG, ASSIST_MODEL_CHECKPOINT, CLASSES, OVERLAY_COLORS, ANNOTATION_ORDER, UI_LANGUAGE

    cfg_path = _resource_path(SETTINGS_FILE)
    data = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: failed to load settings file {cfg_path}: {e}")

    sam_ckpt = data.get("sam_checkpoint", SAM2_CHECKPOINT)
    sam_cfg = data.get("sam_config", SAM2_CONFIG)
    assist_ckpt = data.get("assist_checkpoint", ASSIST_MODEL_CHECKPOINT)
    classes_data = data.get("classes", CLASSES)
    ui_lang = data.get("ui_language", UI_LANGUAGE)
    order_data = data.get("annotation_order", None)

    # 默认的名称与颜色（来自当前脚本）
    default_names = CLASSES
    default_colors = list(OVERLAY_COLORS)

    def _fallback_color(i: int):
        """为新增的 class 生成一个相对分散的颜色，方便区分。"""
        if i < len(default_colors):
            return default_colors[i]
        palette = [
            (1.0, 0.0, 0.0, 0.6),
            (0.0, 1.0, 0.0, 0.6),
            (0.0, 0.0, 1.0, 0.6),
            (1.0, 1.0, 0.0, 0.6),
            (1.0, 0.0, 1.0, 0.6),
            (0.0, 1.0, 1.0, 0.6),
        ]
        j = max(0, i - len(default_colors))
        return palette[j % len(palette)]

    names = list(default_names)
    colors = list(default_colors)

    if isinstance(classes_data, list) and classes_data:
        # 兼容老格式：["Liver", "Fat", ...]
        if all(isinstance(x, str) for x in classes_data):
            names = list(classes_data)
            colors = []
            for i in range(len(names)):
                colors.append(default_colors[i] if i < len(default_colors) else _fallback_color(i))
        # 新格式：[{"name": ..., "color": [r,g,b,a]}, ...]
        elif all(isinstance(x, dict) for x in classes_data):
            names = []
            colors = []
            for i, item in enumerate(classes_data):
                n = item.get("name", default_names[i] if i < len(default_names) else f"Class {i}")
                col = item.get("color", None)
                if isinstance(col, list) and len(col) == 4:
                    try:
                        r, g, b, a = [float(v) for v in col]
                        colors.append((r, g, b, a))
                    except Exception:
                        colors.append(default_colors[i] if i < len(default_colors) else _fallback_color(i))
                else:
                    colors.append(default_colors[i] if i < len(default_colors) else _fallback_color(i))
                names.append(n)

    CLASSES = names
    OVERLAY_COLORS = colors

    # 标注顺序：如果文件里有，就用文件里的；否则用脚本默认，再补齐新增类
    if isinstance(order_data, list) and all(isinstance(x, int) for x in order_data):
        order = [idx for idx in order_data if 0 <= idx < len(CLASSES)]
    else:
        order = list(ANNOTATION_ORDER) if ANNOTATION_ORDER else list(range(len(CLASSES)))
    for idx in range(len(CLASSES)):
        if idx not in order:
            order.append(idx)
    ANNOTATION_ORDER = order

    SAM2_CHECKPOINT = sam_ckpt
    SAM2_CONFIG = sam_cfg
    ASSIST_MODEL_CHECKPOINT = assist_ckpt
    UI_LANGUAGE = ui_lang

    # 回写（保证文件始终可用，且统一为新格式）
    _save_settings()


def _save_settings():
    """将当前全局可编辑设置保存到 JSON 文件中。"""
    cfg_path = _resource_path(SETTINGS_FILE)
    classes_payload = []
    for i, name in enumerate(CLASSES):
        if i < len(OVERLAY_COLORS):
            r, g, b, a = OVERLAY_COLORS[i]
        else:
            r, g, b, a = (1.0, 1.0, 1.0, 0.6)
        classes_payload.append(
            {
                "name": name,
                "color": [float(r), float(g), float(b), float(a)],
            }
        )

    data = {
        "sam_checkpoint": SAM2_CHECKPOINT,
        "sam_config": SAM2_CONFIG,
        "assist_checkpoint": ASSIST_MODEL_CHECKPOINT,
        "classes": classes_payload,
        "annotation_order": ANNOTATION_ORDER,
        "ui_language": UI_LANGUAGE,
    }
    try:
        os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: failed to save settings file {cfg_path}: {e}")


def open_settings_dialog(fig):
    """
    打开一个简单的设置窗口：
      - 左侧 classes 列表，可修改名称；
      - 右侧 SAM checkpoint / config、assist checkpoint 路径可编辑，可浏览；
    保存后写回 JSON，并更新当前全局变量。
    """
    if tk is None or ttk is None:
        print("Settings UI not available (tkinter not installed).")
        return

    try:
        root = fig.canvas.get_tk_widget().winfo_toplevel()
    except Exception:
        print("Settings UI not available (cannot access Tk root).")
        return

    win = tk.Toplevel(root)
    win.title("DeepSurg Settings")
    # 让窗口根据控件内容自动决定合适大小，并设置为最小尺寸，避免每次都要手动放大
    win.transient(root)

    # 文案字典（仅影响 UI，不改变真实类别/模型名称）
    lang = UI_LANGUAGE or "en"
    if lang == "zh":
        texts = {
            "classes_frame": "Classes（可在左侧栏使用，可增删改名与颜色）",
            "selected_name": "选中类名称：",
            "color_frame": "颜色（RGB / 透明度）",
            "add_class": "新增类",
            "delete_class": "删除当前",
            "apply": "应用到当前项",
            "model_paths": "模型路径（重启或下一张图时生效）",
            "sam_ckpt": "SAM ckpt",
            "sam_cfg": "SAM config",
            "assist_ckpt": "Assist ckpt",
            "save": "保存",
            "cancel": "取消",
            "del_warn_title": "提示",
            "del_warn_msg": "至少保留一个类别。",
            "saved_msg": "设置已保存。新的模型路径会在下一张图或下次启动时生效。",
            "contact": "如有问题请联系开发人员 Zechang （xuezechang@gmail.com）",
            "lang_label": "界面语言：",
            "lang_en": "English",
            "lang_zh": "简体中文",
        }
    else:
        texts = {
            "classes_frame": "Classes (available in left panel, editable names & colors)",
            "selected_name": "Selected class name:",
            "color_frame": "Color (RGB / Alpha)",
            "add_class": "Add class",
            "delete_class": "Delete selection",
            "apply": "Apply to selection",
            "model_paths": "Model paths (apply on next image or restart)",
            "sam_ckpt": "SAM ckpt",
            "sam_cfg": "SAM config",
            "assist_ckpt": "Assist ckpt",
            "save": "Save",
            "cancel": "Cancel",
            "del_warn_title": "Warning",
            "del_warn_msg": "At least one class must remain.",
            "saved_msg": "Settings saved. New model paths will take effect for the next image or on next launch.",
            "contact": "For questions please contact developer Zechang (xuezechang@gmail.com).",
            "lang_label": "Language:",
            "lang_en": "English",
            "lang_zh": "简体中文",
        }

    # 顶部语言切换
    top_bar = ttk.Frame(win)
    top_bar.pack(fill="x", padx=10, pady=(8, 0))
    ttk.Label(top_bar, text=texts["lang_label"]).pack(side="left")
    lang_var = tk.StringVar(value="zh" if lang == "zh" else "en")
    lang_combo = ttk.Combobox(
        top_bar,
        textvariable=lang_var,
        state="readonly",
        values=["en", "zh"],
        width=6,
    )
    lang_combo.pack(side="left", padx=(4, 0))
    ttk.Label(top_bar, text=f"({texts['lang_en']}/ {texts['lang_zh']})").pack(side="left", padx=(4, 0))

    # 联系方式提示行
    contact_bar = ttk.Frame(win)
    contact_bar.pack(fill="x", padx=10, pady=(4, 0))
    ttk.Label(contact_bar, text=texts["contact"], foreground="#555555").pack(side="left")

    # ---- Classes 编辑区 ----
    classes_frame = ttk.LabelFrame(win, text=texts["classes_frame"])
    classes_frame.pack(fill="both", expand=True, padx=10, pady=8)

    listbox = tk.Listbox(classes_frame, exportselection=False)
    listbox.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)

    scrollbar = ttk.Scrollbar(classes_frame, orient="vertical", command=listbox.yview)
    scrollbar.pack(side="left", fill="y", pady=8)
    listbox.configure(yscrollcommand=scrollbar.set)

    right_frame = ttk.Frame(classes_frame)
    right_frame.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

    ttk.Label(right_frame, text=texts["selected_name"]).pack(anchor="w")
    name_var = tk.StringVar()
    name_entry = ttk.Entry(right_frame, textvariable=name_var)
    name_entry.pack(fill="x", pady=(0, 6))

    # 颜色（0-255 RGB + 透明度0-100），带预览
    color_frame = ttk.LabelFrame(right_frame, text=texts["color_frame"])
    color_frame.pack(fill="x", pady=(0, 6))

    r_var = tk.IntVar(value=255)
    g_var = tk.IntVar(value=255)
    b_var = tk.IntVar(value=255)
    a_var = tk.IntVar(value=60)  # 0-100 -> 0-1

    def _clamp_int(v, lo, hi):
        return max(lo, min(hi, int(v)))

    def add_rgb_row(parent, label, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=3).pack(side="left")
        scale = ttk.Scale(row, from_=0, to=255, orient="horizontal",
                          command=lambda v, sv=var: sv.set(int(float(v))))
        scale.pack(side="left", fill="x", expand=True, padx=(2, 2))
        entry = ttk.Entry(row, width=4, textvariable=var)
        entry.pack(side="left")

        def on_change(*_):
            try:
                v = int(var.get())
            except Exception:
                v = 0
            v = _clamp_int(v, 0, 255)
            var.set(v)
            scale.set(v)
            update_preview()

        var.trace_add("write", on_change)

    add_rgb_row(color_frame, "R", r_var)
    add_rgb_row(color_frame, "G", g_var)
    add_rgb_row(color_frame, "B", b_var)

    row_a = ttk.Frame(color_frame)
    row_a.pack(fill="x", pady=1)
    ttk.Label(row_a, text="A%", width=3).pack(side="left")
    a_scale = ttk.Scale(row_a, from_=0, to=100, orient="horizontal",
                        command=lambda v: a_var.set(int(float(v))))
    a_scale.pack(side="left", fill="x", expand=True, padx=(2, 2))
    a_entry = ttk.Entry(row_a, width=4, textvariable=a_var)
    a_entry.pack(side="left")

    def on_a_change(*_):
        try:
            v = int(a_var.get())
        except Exception:
            v = 0
        v = _clamp_int(v, 0, 100)
        a_var.set(v)
        a_scale.set(v)
        update_preview()

    a_var.trace_add("write", on_a_change)

    preview = tk.Canvas(right_frame, width=80, height=30, bd=1, relief="sunken")
    preview.pack(pady=(4, 2))

    def update_preview():
        r = _clamp_int(r_var.get(), 0, 255)
        g = _clamp_int(g_var.get(), 0, 255)
        b = _clamp_int(b_var.get(), 0, 255)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        preview.delete("all")
        preview.create_rectangle(0, 0, 80, 30, fill=hex_color, outline="black")

    def refresh_list():
        listbox.delete(0, tk.END)
        for i, name in enumerate(CLASSES):
            listbox.insert(tk.END, f"{i}: {name}")

    def on_select(event=None):
        idxs = listbox.curselection()
        if not idxs:
            return
        idx = idxs[0]
        if 0 <= idx < len(CLASSES):
            name_var.set(CLASSES[idx])
            if idx < len(OVERLAY_COLORS):
                r, g, b, a = OVERLAY_COLORS[idx]
                try:
                    r_var.set(int(float(r) * 255))
                    g_var.set(int(float(g) * 255))
                    b_var.set(int(float(b) * 255))
                    a_var.set(int(float(a) * 100))
                except Exception:
                    pass
            update_preview()

    listbox.bind("<<ListboxSelect>>", on_select)

    def apply_current():
        """将右侧名称与颜色应用到当前选中的 class。"""
        idxs = listbox.curselection()
        if not idxs:
            return
        idx = idxs[0]
        new_name = name_var.get().strip()
        if new_name:
            CLASSES[idx] = new_name
        r = _clamp_int(r_var.get(), 0, 255) / 255.0
        g = _clamp_int(g_var.get(), 0, 255) / 255.0
        b = _clamp_int(b_var.get(), 0, 255) / 255.0
        a = _clamp_int(a_var.get(), 0, 100) / 100.0
        if idx < len(OVERLAY_COLORS):
            OVERLAY_COLORS[idx] = (r, g, b, a)
        else:
            OVERLAY_COLORS.append((r, g, b, a))
        refresh_list()
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(idx)
        listbox.see(idx)

    def add_class():
        """在列表末尾新增一个类，名称和颜色可以后续修改。"""
        idx = len(CLASSES)
        CLASSES.append(f"Class {idx}")
        if idx < len(OVERLAY_COLORS):
            pass
        else:
            OVERLAY_COLORS.append((1.0, 1.0, 1.0, 0.6))
        # 新类加入标注顺序末尾，保证左侧工具栏里能选到
        if idx not in ANNOTATION_ORDER:
            ANNOTATION_ORDER.append(idx)
        refresh_list()
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(idx)
        listbox.see(idx)
        on_select()

    def remove_class():
        """删除当前选中的类，并调整标注顺序索引。"""
        idxs = listbox.curselection()
        if not idxs:
            return
        idx = idxs[0]
        if len(CLASSES) <= 1:
            if messagebox is not None:
                try:
                    messagebox.showwarning(texts["del_warn_title"], texts["del_warn_msg"])
                except Exception:
                    pass
            return
        CLASSES.pop(idx)
        if idx < len(OVERLAY_COLORS):
            OVERLAY_COLORS.pop(idx)
        # 调整 ANNOTATION_ORDER：去掉删除的索引，大于它的全部减一
        new_order = []
        for k in ANNOTATION_ORDER:
            if k == idx:
                continue
            if k > idx:
                new_order.append(k - 1)
            else:
                new_order.append(k)
        ANNOTATION_ORDER.clear()
        ANNOTATION_ORDER.extend(new_order)
        refresh_list()
        listbox.selection_clear(0, tk.END)
        if idx > 0:
            sel = idx - 1
        else:
            sel = 0
        if CLASSES:
            listbox.selection_set(sel)
            listbox.see(sel)
            on_select()

    btn_row = ttk.Frame(right_frame)
    btn_row.pack(fill="x", pady=(2, 0))
    btn_add = ttk.Button(btn_row, text=texts["add_class"], command=add_class)
    btn_add.pack(side="left")
    btn_del = ttk.Button(btn_row, text=texts["delete_class"], command=remove_class)
    btn_del.pack(side="left", padx=(4, 0))
    btn_apply = ttk.Button(btn_row, text=texts["apply"], command=apply_current)
    btn_apply.pack(side="right")

    refresh_list()
    if CLASSES:
        listbox.selection_set(0)
        on_select()

    # 让布局先完成，再根据实际需要的宽高设置窗口尺寸和最小尺寸
    try:
        win.update_idletasks()
        req_w = win.winfo_reqwidth()
        req_h = win.winfo_reqheight()
        # 给一个更大的下限，避免在高分屏/大字体下内容被截断
        min_w = max(req_w, 1100)
        min_h = max(req_h, 650)
        win.geometry(f"{min_w}x{min_h}")
        win.minsize(min_w, min_h)
        win.resizable(True, True)
    except Exception:
        pass

    # ---- 模型路径区 ----
    models_frame = ttk.LabelFrame(win, text=texts["model_paths"])
    models_frame.pack(fill="x", padx=10, pady=(0, 8))

    sam_ckpt_var = tk.StringVar(value=SAM2_CHECKPOINT)
    sam_cfg_var = tk.StringVar(value=SAM2_CONFIG)
    assist_ckpt_var = tk.StringVar(value=ASSIST_MODEL_CHECKPOINT)

    def add_path_row(parent, label, var, filetypes):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=14).pack(side="left")
        entry = ttk.Entry(row, textvariable=var)
        entry.pack(side="left", fill="x", expand=True, padx=(0, 4))

        def browse():
            if filedialog is None:
                return
            path = filedialog.askopenfilename(parent=win, filetypes=filetypes)
            if path:
                var.set(os.path.relpath(path, os.getcwd()))

        ttk.Button(row, text="浏览...", command=browse).pack(side="left")

    add_path_row(models_frame, texts["sam_ckpt"], sam_ckpt_var, [("Checkpoint", "*.pt;*.pth;*.bin"), ("All files", "*.*")])
    add_path_row(models_frame, texts["sam_cfg"], sam_cfg_var, [("YAML", "*.yaml;*.yml"), ("All files", "*.*")])
    add_path_row(models_frame, texts["assist_ckpt"], assist_ckpt_var, [("Checkpoint", "*.ckpt;*.pt;*.pth;*.bin"), ("All files", "*.*")])

    # ---- 底部按钮 ----
    btn_frame = ttk.Frame(win)
    btn_frame.pack(fill="x", padx=10, pady=(0, 10))

    def on_save():
        global SAM2_CHECKPOINT, SAM2_CONFIG, ASSIST_MODEL_CHECKPOINT, UI_LANGUAGE, _settings_dirty, _assist_model_cache
        # 先把当前右侧面板的名称和颜色应用到选中的类，避免用户忘记点“应用到当前项”时丢失修改
        try:
            apply_current()
        except Exception:
            pass
        # 保存界面语言，不影响 classes / 模型本身
        UI_LANGUAGE = lang_var.get() or "en"
        SAM2_CHECKPOINT = sam_ckpt_var.get().strip() or SAM2_CHECKPOINT
        SAM2_CONFIG = sam_cfg_var.get().strip() or SAM2_CONFIG
        ASSIST_MODEL_CHECKPOINT = assist_ckpt_var.get().strip() or ASSIST_MODEL_CHECKPOINT
        # 标记设置已更改：下一张图前重载模型 / assist 模型
        _settings_dirty = True
        _assist_model_cache = None
        _save_settings()
        # 立即刷新主标注窗口 UI 文案（按钮和状态栏），无需等到下一张图
        _refresh_main_ui_language(fig)
        if messagebox is not None:
            try:
                messagebox.showinfo("Settings", texts["saved_msg"])
            except Exception:
                pass
        win.destroy()

    def on_cancel():
        win.destroy()

    btn_save = ttk.Button(btn_frame, text=texts["save"], command=on_save)
    btn_save.pack(side="right", padx=(4, 0))
    btn_cancel = ttk.Button(btn_frame, text=texts["cancel"], command=on_cancel)
    btn_cancel.pack(side="right")

    # 语言切换时，动态更新当前窗口内的文字
    def on_lang_change(event=None):
        val = lang_var.get()
        new_lang = "zh" if val == "zh" else "en"
        nonlocal_texts = {}
        if new_lang == "zh":
            nonlocal_texts.update({
                "classes_frame": "Classes（可在左侧栏使用，可增删改名与颜色）",
                "selected_name": "选中类名称：",
                "color_frame": "颜色（RGB / 透明度）",
                "add_class": "新增类",
                "delete_class": "删除当前",
                "apply": "应用到当前项",
                "model_paths": "模型路径（重启或下一张图时生效）",
                "save": "保存",
                "cancel": "取消",
            })
        else:
            nonlocal_texts.update({
                "classes_frame": "Classes (available in left panel, editable names & colors)",
                "selected_name": "Selected class name:",
                "color_frame": "Color (RGB / Alpha)",
                "add_class": "Add class",
                "delete_class": "Delete selection",
                "apply": "Apply to selection",
                "model_paths": "Model paths (apply on next image or restart)",
                "save": "Save",
                "cancel": "Cancel",
            })
        try:
            classes_frame.configure(text=nonlocal_texts["classes_frame"])
            for child in right_frame.winfo_children():
                # 第一个 Label 是 selected_name
                if isinstance(child, ttk.Label):
                    child.configure(text=nonlocal_texts["selected_name"])
                    break
            color_frame.configure(text=nonlocal_texts["color_frame"])
            btn_add.configure(text=nonlocal_texts["add_class"])
            btn_del.configure(text=nonlocal_texts["delete_class"])
            btn_apply.configure(text=nonlocal_texts["apply"])
            models_frame.configure(text=nonlocal_texts["model_paths"])
            btn_save.configure(text=nonlocal_texts["save"])
            btn_cancel.configure(text=nonlocal_texts["cancel"])
        except Exception:
            pass

    lang_combo.bind("<<ComboboxSelected>>", on_lang_change)

    win.grab_set()
    win.focus_set()


def _on_close(event):
    """Handle figure close: set flag and stop event loop so main loop can skip to next image."""
    global _window_closed
    _window_closed = True
    try:
        if event is not None and hasattr(event, "canvas") and event.canvas is not None:
            event.canvas.stop_event_loop()
    except Exception:
        pass


def _get_device():
    """Use GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model():
    """Load SAM2 model and image predictor. Returns (predictor, device)."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        print(
            "Error: Cannot import sam2. Install SAM2 in this project (e.g. pip install -e .).\n"
            f"Details: {e}"
        )
        sys.exit(1)

    device = _get_device()
    print(f"Using device: {device}")

    # Resolve checkpoint/config paths so they work both in source and bundled .exe
    ckpt_path = _resource_path(SAM2_CHECKPOINT)
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Set SAM2_CHECKPOINT at the top of this script to your sam2.1_hiera_large.pt or sam2.1_hiera_small.pt path.")
        sys.exit(1)

    # 配置文件有两份拷贝：
    # - 项目根目录下的 configs/sam2.1/sam2.1_hiera_l.yaml（便于你查看/修改）
    # - sam2 包内部的 configs（Hydra 在这里按“配置名”查找）
    # 这里使用绝对路径仅用于日志输出，而真正传给 build_sam2 的是配置名本身，
    # 让 Hydra 在 sam2 包内部的配置目录中查找，避免不同操作系统上的路径差异。
    config_path = _resource_path(SAM2_CONFIG)
    config_name = SAM2_CONFIG.replace("\\", "/")  # "configs/sam2.1/sam2.1_hiera_l.yaml"
    print(f"Loading SAM2: config={config_path}, checkpoint={ckpt_path}")
    sam_model = build_sam2(config_file=config_name, ckpt_path=ckpt_path, device=str(device))
    predictor = SAM2ImagePredictor(
        sam_model,
        max_sprinkle_area=SAM2_MAX_SPRINKLE_AREA,
        max_hole_area=SAM2_MAX_HOLE_AREA,
    )
    print("SAM2 model loaded.")
    return predictor, device


def _sample_stroke_to_points(stroke_xy, interval_pixels):
    """Sample stroke trajectory at interval for SAM point prompts."""
    if len(stroke_xy) < 2:
        return stroke_xy
    out = [stroke_xy[0]]
    cum = 0.0
    for i in range(1, len(stroke_xy)):
        dx = stroke_xy[i][0] - stroke_xy[i - 1][0]
        dy = stroke_xy[i][1] - stroke_xy[i - 1][1]
        cum += (dx * dx + dy * dy) ** 0.5
        while cum >= interval_pixels and interval_pixels > 0:
            cum -= interval_pixels
            t = 1.0 - cum / interval_pixels if interval_pixels > 0 else 1.0
            t = max(0, min(1, t))
            nx = stroke_xy[i - 1][0] + t * (stroke_xy[i][0] - stroke_xy[i - 1][0])
            ny = stroke_xy[i - 1][1] + t * (stroke_xy[i][1] - stroke_xy[i - 1][1])
            out.append((float(nx), float(ny)))
    out.append(stroke_xy[-1])
    return out


def get_click_points(ax, fig, image_shape, model_assist_ax=None):
    """
    Add foreground prompts: click/drag then ENTER; or click "Model assist" button.
    model_assist_ax: if given and ASSIST_MODEL_ENABLED, a "Model assist" button is shown.
    Returns:
      - list of (x, y) points for SAM prompts,
      - [] (ESC skip),
      - None (window closed),
      - ("model_assist",) when the model-assist button is clicked,
      - ("polygon", stroke_xy) when the user draws a closed freehand region (used as direct mask).
    """
    global _window_closed
    _window_closed = False
    points = []
    pts_display = []
    stroke_line = None
    dragging = False
    stroke_points = []
    model_assist_requested = [False]
    freehand_polygon = [None]

    def in_axes(e):
        return e.inaxes == ax and e.button == 1 and e.xdata is not None and e.ydata is not None

    def on_press(event):
        nonlocal dragging, stroke_points, stroke_line
        if not in_axes(event):
            return
        dragging = True
        stroke_points = [(float(event.xdata), float(event.ydata))]
        stroke_points_ref = [stroke_points[0]]
        ax._current_stroke = stroke_points_ref
        stroke_line = ax.plot(
            [stroke_points_ref[0][0]], [stroke_points_ref[0][1]],
            color="lime", linewidth=2, alpha=0.9, solid_capstyle="round"
        )[0]
        fig.canvas.draw_idle()

    def on_motion(event):
        if not dragging or event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if not hasattr(ax, "_current_stroke"):
            return
        ax._current_stroke.append((float(event.xdata), float(event.ydata)))
        if stroke_line is not None and len(ax._current_stroke) >= 1:
            xs = [p[0] for p in ax._current_stroke]
            ys = [p[1] for p in ax._current_stroke]
            stroke_line.set_data(xs, ys)
        fig.canvas.draw_idle()

    def on_release(event):
        nonlocal dragging, stroke_points, stroke_line
        if event.button != 1:
            return
        if not dragging:
            return
        dragging = False
        stroke = getattr(ax, "_current_stroke", [])
        if hasattr(ax, "_current_stroke"):
            del ax._current_stroke
        # Click vs drag
        if len(stroke) >= 2:
            length = 0.0
            for i in range(1, len(stroke)):
                dx = stroke[i][0] - stroke[i - 1][0]
                dy = stroke[i][1] - stroke[i - 1][1]
                length += (dx * dx + dy * dy) ** 0.5
            if length >= STROKE_MIN_LENGTH_PIXELS:
                # Check if stroke roughly closes back to the start: treat as freehand region
                dx0 = stroke[-1][0] - stroke[0][0]
                dy0 = stroke[-1][1] - stroke[0][1]
                if (dx0 * dx0 + dy0 * dy0) ** 0.5 <= FREEHAND_CLOSE_DISTANCE_PIXELS:
                    freehand_polygon[0] = list(stroke)
                    try:
                        fig.canvas.stop_event_loop()
                    except Exception:
                        pass
                    return
                # Otherwise, treat as dense stroke for SAM prompts
                sampled = _sample_stroke_to_points(stroke, STROKE_SAMPLE_INTERVAL_PIXELS)
                points.extend(sampled)
                # stroke_line already shows the path, keep it
                fig.canvas.draw_idle()
                return
        # Short drag or click: single point
        if stroke:
            points.append((float(stroke[0][0]), float(stroke[0][1])))
            ax.plot(stroke[0][0], stroke[0][1], "o", color="lime", markersize=8, markeredgecolor="white", markeredgewidth=1)
        if stroke_line is not None:
            try:
                stroke_line.remove()
            except Exception:
                pass
        fig.canvas.draw_idle()

    esc_skip = [False]

    def onkey(event):
        if event.key in ("enter", "return"):
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass
        elif event.key == "escape":
            esc_skip[0] = True
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid_press = fig.canvas.mpl_connect("button_press_event", on_press)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", on_motion)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_release)
    cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    fig.canvas.mpl_connect("close_event", _on_close)

    btn_widget = None
    if model_assist_ax is not None and ASSIST_MODEL_ENABLED:
        model_assist_ax.clear()
        btn_widget = mwidgets.Button(model_assist_ax, _ui_text("model_assist_button"))
        # 记录引用，语言切换时可以统一刷新按钮文字
        try:
            fig._model_assist_btn = btn_widget
        except Exception:
            pass

        def on_model_assist_clicked(event):
            model_assist_requested[0] = True
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

        btn_widget.on_clicked(on_model_assist_clicked)

    ax.set_title(_ui_text("click_or_drag"))
    fig.canvas.draw()
    try:
        fig.canvas.start_event_loop(timeout=-1)
    except Exception:
        pass

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_key)

    if _window_closed:
        return None
    if model_assist_requested[0]:
        return ("model_assist",)
    if freehand_polygon[0] is not None:
        return ("polygon", freehand_polygon[0])
    if esc_skip[0]:
        return []
    return points if points else None


def _update_status_line(fig, class_name, prompt_line, hint=""):
    """Bottom status line: current class + prompt + key hint."""
    if getattr(fig, "_status_text", None) is None:
        return
    if hint:
        s = f"{_ui_text('status_current_prefix')}{class_name or ''}  |  {prompt_line or ''}  |  {hint}"
    else:
        s = f"{_ui_text('status_current_prefix')}{class_name or ''}  |  {prompt_line or ''}"
    fig._status_text.set_text(s)
    fig._status_text.set_fontsize(10)


def wait_yes_no_in_window(fig, class_name, prompt_line):
    """Wait for key in window: ENTER=yes, ESC=no. Returns 'y'/'n'/None if closed."""
    global _window_closed
    _update_status_line(fig, class_name, prompt_line)
    fig.canvas.draw()
    result = [None]

    def onkey(event):
        if event.key in ("enter", "return"):
            result[0] = "y"
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass
        elif event.key == "escape":
            result[0] = "n"
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid = fig.canvas.mpl_connect("key_press_event", onkey)
    try:
        fig.canvas.start_event_loop(timeout=-1)
    except Exception:
        pass
    fig.canvas.mpl_disconnect(cid)
    if _window_closed:
        return None
    return result[0]


def _smooth_mask(mask_binary, min_component_pixels):
    """Remove small components, fill holes, then smooth jagged edges (closing)."""
    try:
        from scipy import ndimage
    except ImportError:
        return mask_binary
    mask = (mask_binary > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask_binary
    # Remove small components
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask_binary
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    keep = (sizes >= min_component_pixels).astype(np.int32)
    keep = np.concatenate([[0], keep])
    mask_cleaned = keep[labeled]
    mask_cleaned = (mask_cleaned > 0).astype(np.uint8)
    # Fill holes
    mask_cleaned = ndimage.binary_fill_holes(mask_cleaned).astype(np.uint8)
    # Smooth jagged boundary: morphological closing with small disk
    if SMOOTH_EDGE_RADIUS > 0:
        r = SMOOTH_EDGE_RADIUS
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        disk = ((x * x + y * y) <= (r * r)).astype(np.uint8)
        mask_cleaned = ndimage.binary_closing(mask_cleaned, structure=disk).astype(np.uint8)
    return mask_cleaned


def _keep_only_components_containing_points(mask_binary, point_coords_xy):
    """
    Keep only connected components that contain at least one user point.
    Removes "leaked" regions far from the click/stroke (e.g. lower-right when user drew in upper-left).
    """
    try:
        from scipy import ndimage
    except ImportError:
        return mask_binary
    mask = (mask_binary > 0).astype(np.uint8)
    if mask.sum() == 0 or point_coords_xy is None or len(point_coords_xy) == 0:
        return mask_binary
    h, w = mask.shape
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask_binary
    # Which component(s) contain at least one of the user's points?
    keep_labels = set()
    for (x, y) in point_coords_xy:
        col = int(round(float(x)))
        row = int(round(float(y)))
        col = max(0, min(w - 1, col))
        row = max(0, min(h - 1, row))
        L = labeled[row, col]
        if L > 0:
            keep_labels.add(L)
    if not keep_labels:
        # No point fell on mask: keep the component closest to the centroid of user points
        cy = np.mean([float(y) for (x, y) in point_coords_xy])
        cx = np.mean([float(x) for (x, y) in point_coords_xy])
        row = max(0, min(h - 1, int(round(cy))))
        col = max(0, min(w - 1, int(round(cx))))
        best_label = None
        best_d = np.inf
        for L in range(1, num_features + 1):
            comp = (labeled == L)
            dist_to_comp = ndimage.distance_transform_edt(~comp)
            d = float(dist_to_comp[row, col])
            if d < best_d:
                best_d = d
                best_label = L
        if best_label is not None:
            keep_labels.add(best_label)
    out = np.zeros_like(mask)
    for L in keep_labels:
        out[labeled == L] = 1
    return out.astype(np.uint8)


def postprocess_combined_mask(combined_mask, num_classes):
    """
    Final per-class post-processing applied once on the combined multi-class mask
    right before saving:
      - remove tiny speckles per class
      - fill small holes per class
      - smooth jagged boundaries with a small disk, using per-class radius.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return combined_mask

    out = combined_mask.copy().astype(np.int32)
    max_label = int(out.max())
    if max_label <= 0:
        return combined_mask

    h, w = out.shape

    for cls_idx in range(num_classes):
        label_val = cls_idx + 1
        if label_val > max_label:
            continue

        # Binary mask for this class
        mask = (out == label_val).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # Per-class parameters, with sane defaults
        min_pixels, radius = CLASS_POSTPROCESS_CONFIG.get(
            cls_idx, (MIN_MASK_COMPONENT_PIXELS, SMOOTH_EDGE_RADIUS)
        )
        if min_pixels <= 0 and radius <= 0:
            continue

        # Remove small components
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        keep = (sizes >= max(1, int(min_pixels))).astype(np.int32)
        keep = np.concatenate([[0], keep])
        mask_cleaned = keep[labeled]
        mask_cleaned = (mask_cleaned > 0).astype(np.uint8)

        # Fill small holes
        mask_cleaned = ndimage.binary_fill_holes(mask_cleaned).astype(np.uint8)

        # Smooth jagged edge with a per-class radius
        if radius > 0:
            r = int(radius)
            y, x = np.ogrid[-r : r + 1, -r : r + 1]
            disk = ((x * x + y * y) <= (r * r)).astype(np.uint8)
            mask_cleaned = ndimage.binary_closing(mask_cleaned, structure=disk).astype(
                np.uint8
            )

        # Update only this class's region in the output; other labels untouched.
        out[out == label_val] = 0
        out[mask_cleaned > 0] = label_val

    return np.clip(out, 0, 255).astype(np.uint8)


def generate_mask(predictor, image_rgb, point_coords_xy, multimask_output=True):
    """
    Generate binary mask from foreground point prompts using SAM,
    then smooth it and keep only the region(s) that contain the user's points (no leakage).
    """
    if point_coords_xy is None or len(point_coords_xy) == 0:
        return None

    coords = np.array(point_coords_xy, dtype=np.float32)
    labels = np.ones(len(coords), dtype=np.int32)  # 1 = foreground

    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=multimask_output,
        return_logits=False,
    )

    if len(masks) > 1:
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
    else:
        mask = masks[0]

    mask = mask.astype(np.uint8)
    mask = _smooth_mask(mask, MIN_MASK_COMPONENT_PIXELS)
    mask = _keep_only_components_containing_points(mask, point_coords_xy)
    return mask


def overlay_mask(ax, image_rgb, mask_binary, color_rgba, title=None, show_edge=False):
    """Overlay binary mask on image. show_edge=True draws a bright contour."""
    ax.clear()
    ax.imshow(image_rgb)
    if mask_binary is not None and mask_binary.size > 0:
        overlay = np.zeros((*mask_binary.shape[:2], 4), dtype=np.float32)
        mask_bool = mask_binary > 0
        overlay[mask_bool] = color_rgba
        ax.imshow(overlay)
        if show_edge and np.any(mask_bool):
            # Contour at 0.5
            ax.contour(mask_binary.astype(np.float32), levels=[0.5], colors=[PREVIEW_EDGE_COLOR[:3]], linewidths=2)
    if title:
        ax.set_title(title)
    ax.axis("off")


def _overlay_boundary(ax, boundary_mask):
    """Draw user-defined boundary lines on top of the current image/mask view."""
    if boundary_mask is None:
        return
    if not np.any(boundary_mask):
        return
    try:
        ax.contour(
            boundary_mask.astype(np.float32),
            levels=[0.5],
            colors=[(1.0, 1.0, 0.0)],
            linewidths=1.5,
        )
    except Exception:
        # Overlay is purely visual; ignore failures to avoid breaking annotation flow.
        pass


def draw_boundary_line(ax, fig, boundary_mask):
    """
    Let the user define a boundary line with the left mouse button.

    Two interaction styles are supported in the same tool:
      - Drag with left button to draw a freehand stroke.
      - Single left-click multiple times to add vertices and draw line segments between them.

    Right-click undoes the last vertex / sample point.
    Press ENTER to finish and ESC to cancel.

    The stroke is written into boundary_mask (thickened by BOUNDARY_LINE_WIDTH_PIXELS).
    Returns the updated boundary_mask.
    """
    import cv2

    if boundary_mask is None:
        raise ValueError("boundary_mask must be initialized before calling draw_boundary_line")

    h, w = boundary_mask.shape[:2]
    stroke = []  # list of (x, y) in data coordinates, collected from drag or clicks
    drawing = {"active": False}  # True only while left button is pressed (for drag)
    is_dragging = {"value": False}  # True once mouse has moved with button held down
    line_artist = {"obj": None}
    esc_cancel = {"value": False}

    def in_axes(event):
        return (
            event.inaxes == ax
            and event.xdata is not None
            and event.ydata is not None
        )

    def redraw():
        # Remove old artist if any
        if line_artist["obj"] is not None:
            try:
                line_artist["obj"].remove()
            except Exception:
                pass
            line_artist["obj"] = None

        if len(stroke) >= 1:
            xs = [p[0] for p in stroke]
            ys = [p[1] for p in stroke]
            # For drag strokes, show only a smooth polyline (no per-point markers);
            # for multi-click vertices, also show small markers at each vertex.
            if is_dragging["value"]:
                line_artist["obj"] = ax.plot(
                    xs,
                    ys,
                    color="yellow",
                    linewidth=2,
                    alpha=0.9,
                    solid_capstyle="round",
                )[0]
            else:
                line_artist["obj"] = ax.plot(
                    xs,
                    ys,
                    color="yellow",
                    linewidth=2,
                    alpha=0.9,
                    solid_capstyle="round",
                    marker="o",
                    markersize=4,
                    markerfacecolor="yellow",
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )[0]

        fig.canvas.draw_idle()

    def on_press(event):
        if not in_axes(event):
            return
        # Left click: start/extend boundary (drag or multi-click)
        if event.button == 1:
            drawing["active"] = True
            is_dragging["value"] = False  # will be set True once mouse moves
            stroke.append((float(event.xdata), float(event.ydata)))
            redraw()
        # Right click: undo last point
        elif event.button == 3:
            if stroke:
                stroke.pop()
                redraw()

    def on_motion(event):
        if not drawing["active"] or not in_axes(event):
            return
        # When dragging with left button pressed, sample dense points along the stroke
        is_dragging["value"] = True
        stroke.append((float(event.xdata), float(event.ydata)))
        redraw()

    def on_release(event):
        if event.button != 1:
            return
        # Stop drag sampling, but do NOT end the whole interaction;
        # user must press ENTER to finish or ESC to cancel.
        drawing["active"] = False

    def on_key(event):
        # ENTER / RETURN: finish current boundary stroke
        if event.key in ("enter", "return"):
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass
        elif event.key == "escape":
            esc_cancel["value"] = True
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid_press = fig.canvas.mpl_connect("button_press_event", on_press)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", on_motion)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_release)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    try:
        fig.canvas.start_event_loop(timeout=-1)
    except Exception:
        pass

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_key)

    # If user cancelled or stroke too short, keep original mask
    if esc_cancel["value"] or len(stroke) < 2:
        return boundary_mask

    # Write stroke into boundary mask (candidate)
    mask_updated = boundary_mask.copy()
    pts = np.array(stroke, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_int = pts.astype(np.int32)
    for i in range(1, len(pts_int)):
        p0 = (int(pts_int[i - 1, 0]), int(pts_int[i - 1, 1]))
        p1 = (int(pts_int[i, 0]), int(pts_int[i, 1]))
        cv2.line(
            mask_updated,
            p0,
            p1,
            color=1,
            thickness=max(1, int(BOUNDARY_LINE_WIDTH_PIXELS)),
        )

    # Directly accept this boundary line: the first ENTER closes the drawing
    # loop and we apply the boundary immediately (ESC has already been handled).
    # Remove the temporary stroke line from the display if possible.
    if line_artist["obj"] is not None:
        try:
            line_artist["obj"].remove()
        except Exception:
            pass
        fig.canvas.draw_idle()

    return mask_updated


def polygon_click_region(fig, ax, image_shape):
    """
    Let the user select a polygonal region by left-clicking multiple vertices:
      - Left click: add a vertex and draw lines between consecutive points
      - Right click: undo the last vertex
      - ENTER: finish and close polygon (connect last to first)
      - ESC: cancel and return None
    Returns a binary mask (H, W) with 1 inside the polygon, or None if cancelled.
    """
    import cv2

    h, w = image_shape
    points = []
    esc_cancel = {"value": False}
    line_artist = {"obj": None}

    def redraw():
        if line_artist["obj"] is not None:
            try:
                line_artist["obj"].remove()
            except Exception:
                pass
            line_artist["obj"] = None
        if len(points) >= 1:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            # 显示多边形边以及每个顶点位置
            line_artist["obj"] = ax.plot(
                xs,
                ys,
                color="yellow",
                linewidth=2,
                alpha=0.9,
                solid_capstyle="round",
                marker="o",
                markersize=6,
                markerfacecolor="yellow",
                markeredgecolor="black",
                markeredgewidth=0.8,
            )[0]
        fig.canvas.draw_idle()

    def in_axes(event):
        return (
            event.inaxes == ax
            and event.xdata is not None
            and event.ydata is not None
        )

    def on_click(event):
        if not in_axes(event):
            return
        # Left click: add a vertex
        if event.button == 1:
            points.append((float(event.xdata), float(event.ydata)))
            redraw()
        # Right click: undo last vertex
        elif event.button == 3:
            if points:
                points.pop()
                redraw()

    def on_key(event):
        if event.key in ("enter", "return"):
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass
        elif event.key == "escape":
            esc_cancel["value"] = True
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_title(
        "Polygon: left-click vertices, right-click undo, ENTER=finish, ESC=cancel."
    )
    fig.canvas.draw_idle()

    try:
        fig.canvas.start_event_loop(timeout=-1)
    except Exception:
        pass

    fig.canvas.mpl_disconnect(cid_click)
    fig.canvas.mpl_disconnect(cid_key)

    # Clear temporary polyline from display
    if line_artist["obj"] is not None:
        try:
            line_artist["obj"].remove()
        except Exception:
            pass
        fig.canvas.draw_idle()

    if esc_cancel["value"] or len(points) < 3:
        return None

    # Close polygon by connecting last point to first
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    pts_int = pts.astype(np.int32)

    mask_binary = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_binary, [pts_int], 1)
    return mask_binary


# -----------------------------------------------------------------------------
# Model-assisted labeling (optional): trained segmentation model + box prompt
# 与 segmentation/src/predict_masks.py 一致：320x320 + smp encoder 预处理 (0–255)
# -----------------------------------------------------------------------------
_assist_model_cache = None  # (model, device) or None
ASSIST_INPUT_SIZE = 320
# The assist model is trained on the first 13 classes only (indices 0..12).
ASSIST_NUM_CLASSES = 13


def load_assist_model():
    """Lazy-load 13-class FPN; returns (model, device) or None on failure."""
    global _assist_model_cache
    if _assist_model_cache is not None:
        return _assist_model_cache
    if not ASSIST_MODEL_ENABLED:
        return None
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("[Assist] segmentation_models_pytorch not installed; Model assist disabled.")
        return None
    # Resolve checkpoint path so it works both in source and bundled .exe
    ckpt_resolved = _resource_path(ASSIST_MODEL_CHECKPOINT)
    ckpt_path = Path(ckpt_resolved)
    if not ckpt_path.exists():
        print(f"[Assist] Checkpoint not found: {ckpt_resolved}; Model assist disabled.")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.FPN(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=ASSIST_NUM_CLASSES,
        activation=None,
    )
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
            state = {k: v for k, v in state.items() if k not in ("std", "mean")}
            while any(k.startswith("model.") for k in state):
                state = {k[6:] if k.startswith("model.") else k: v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    except Exception as e:
        print(f"[Assist] Failed to load checkpoint: {e}; Model assist disabled.")
        return None
    model.to(device)
    model.eval()
    _assist_model_cache = (model, device)
    return _assist_model_cache


def predict_assist_mask(image_rgb, model, device, class_idx, box_xyxy=None):
    """
    与 predict_masks.py 一致：320x320 输入，smp encoder 预处理 (0–255)，输出取 argmax 得当前类 mask。
    """
    import cv2
    from scipy import ndimage
    import segmentation_models_pytorch as smp
    h, w = image_rgb.shape[:2]
    # Assist model只训练了前 ASSIST_NUM_CLASSES 个类别；超出范围时直接返回空 mask。
    if class_idx is None or class_idx >= ASSIST_NUM_CLASSES:
        return np.zeros((h, w), dtype=np.uint8)
    if box_xyxy is not None:
        x1, y1, x2, y2 = [int(round(t)) for t in box_xyxy]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8)
        img = image_rgb[y1:y2, x1:x2]
    else:
        img = image_rgb
    # 与 predict_masks.py 一致：resize 320x320，0–255 float，smp encoder 预处理
    img_resized = cv2.resize(img, (ASSIST_INPUT_SIZE, ASSIST_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img_resized).float().to(device).unsqueeze(0)
    params = smp.encoders.get_preprocessing_params("efficientnet-b4")
    mean = torch.tensor(params["mean"], device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(params["std"], device=device, dtype=x.dtype).view(1, 3, 1, 1)
    x = x.permute(0, 3, 1, 2)
    x = (x - mean) / std
    with torch.no_grad():
        logits = model(x)
    logits = logits[0]
    ch = min(class_idx, logits.shape[0] - 1)
    pred = (torch.argmax(logits, dim=0) == ch).cpu().numpy().astype(np.uint8)
    if box_xyxy is not None:
        crop_h, crop_w = y2 - y1, x2 - x1
        if (pred.shape[0], pred.shape[1]) != (crop_h, crop_w):
            pred = ndimage.zoom(pred, (crop_h / pred.shape[0], crop_w / pred.shape[1]), order=1)
            pred = (pred > 0.5).astype(np.uint8)
        out = np.zeros((h, w), dtype=np.uint8)
        out[y1:y2, x1:x2] = pred
        return out
    if (pred.shape[0], pred.shape[1]) != (h, w):
        pred = ndimage.zoom(pred, (h / pred.shape[0], w / pred.shape[1]), order=1)
        pred = (pred > 0.5).astype(np.uint8)
    return pred


def get_rectangle(fig, ax):
    """Let user draw a rectangle on the image; return (x1, y1, x2, y2) in data coords or None on cancel."""
    result = [None]
    rect_selector = []

    def on_select(eclick, erelease):
        x1, x2 = min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata)
        y1, y2 = min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata)
        result[0] = (float(x1), float(y1), float(x2), float(y2))
        ax.set_title("Rectangle drawn. Press ENTER to confirm, ESC to cancel.")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "escape":
            result[0] = "cancel"
        if event.key in ("enter", "return") or event.key == "escape":
            try:
                fig.canvas.stop_event_loop()
            except Exception:
                pass

    rs = mwidgets.RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],
        minspanx=5, minspany=5,
        spancoords="data",
        interactive=False,
    )
    rect_selector.append(rs)
    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    ax.set_title("Draw a rectangle (drag on image), then ENTER to confirm, ESC to cancel.")
    fig.canvas.draw()
    # Ensure figure window gets key events (fixes "stuck" when ENTER not received on Windows/TkAgg)
    try:
        w = fig.canvas.get_tk_widget()
        if w is not None:
            w.focus_set()
    except Exception:
        pass
    try:
        fig.canvas.start_event_loop(timeout=-1)
    except Exception:
        pass
    fig.canvas.mpl_disconnect(cid)
    try:
        rs.remove()
    except Exception:
        pass
    if _window_closed or result[0] == "cancel":
        return None
    return result[0]


def fill_gaps_in_combined_mask(combined_mask, num_classes, max_distance=None):
    """
    Assign every unlabeled pixel to the nearest class (Voronoi-style).
    Leaves no unlabeled region: remaining pixels expand to the closest class (needs scipy).
    Never assigns to FILL_GAPS_EXCLUDE_CLASSES (e.g. instruments).
    max_distance is ignored; all label-0 pixels are filled.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return combined_mask
    out = combined_mask.copy().astype(np.int32)
    labeled = (combined_mask > 0)
    if not np.any(labeled):
        return combined_mask
    to_fill = (combined_mask == 0)
    if not np.any(to_fill):
        return np.clip(out, 0, 255).astype(np.uint8)
    # Only consider non-excluded classes when filling (e.g. no instruments)
    fillable = [k for k in range(num_classes) if k not in FILL_GAPS_EXCLUDE_CLASSES]
    if not fillable:
        return np.clip(out, 0, 255).astype(np.uint8)
    dist_stack = np.stack(
        [ndimage.distance_transform_edt(combined_mask != (k + 1)) for k in fillable],
        axis=0,
    )
    best_idx = np.argmin(dist_stack, axis=0)
    out[to_fill] = (np.array(fillable, dtype=out.dtype)[best_idx[to_fill]] + 1).astype(out.dtype)
    return np.clip(out, 0, 255).astype(np.uint8)


def save_mask(combined_mask, save_path):
    """
    Save the combined multi-class mask as PNG (integer labels: 0 = background, 1..N = classes).

    Args:
        combined_mask: (H, W) uint8 or int, values 0..num_classes.
        save_path: path to output file (e.g. masks/<image_name>.png).
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # PNG supports 0..255; we use uint8
    out = np.clip(combined_mask, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(save_path)
    print(f"Saved: {save_path}")


def save_mask_image(image_rgb, combined_mask, save_path):
    """Save visualization (image + mask overlay) to images_mask/."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    h, w = image_rgb.shape[:2]
    out = image_rgb.astype(np.float32) / 255.0
    max_label = int(combined_mask.max())
    for label in range(1, max_label + 1):
        if label > len(OVERLAY_COLORS):
            break
        r, g, b, a = OVERLAY_COLORS[label - 1]
        sel = (combined_mask == label)
        if not np.any(sel):
            continue
        out[sel, 0] = out[sel, 0] * (1 - a) + r * a
        out[sel, 1] = out[sel, 1] * (1 - a) + g * a
        out[sel, 2] = out[sel, 2] * (1 - a) + b * a
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save(save_path)
    print(f"Saved mask image: {save_path}")


def annotate_image(predictor, device, image_path, image_rgb, fig, ax_display, ax_toolbar):
    """
    All interaction in the window: status line at bottom, ENTER=yes, ESC=no.
    Model weights are loaded once at startup; per-image delay is SAM encoding this image.
    Toolbar on the left lets the user choose the active class; a Finish button ends the image.
    """
    global _window_closed
    _window_closed = False
    h, w = image_rgb.shape[:2]

    _update_status_line(fig, "", "Encoding image (SAM)...", "")
    fig.canvas.draw()
    fig.canvas.flush_events()
    predictor.set_image(image_rgb)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    boundary_mask = np.zeros((h, w), dtype=np.uint8)

    # State shared between callbacks and main loop
    current_class = {"idx": ANNOTATION_ORDER[0] if ANNOTATION_ORDER else 0}
    finish_requested = {"value": False}

    # Build left toolbar with class radio buttons (user chooses which class to annotate).
    # 只在第一次调用时创建 RadioButtons，后续图片复用，避免反复清空导致 Matplotlib 内部状态出错。
    if not hasattr(ax_toolbar, "_class_radio"):
        ax_toolbar.clear()
        ax_toolbar.set_title(_ui_text("toolbar_classes"), fontsize=10)
        ax_toolbar.set_xticks([])
        ax_toolbar.set_yticks([])
        # Remove frame around the class list for a cleaner look
        ax_toolbar.set_frame_on(False)
        labels = [CLASSES[c] for c in ANNOTATION_ORDER]
        radio = mwidgets.RadioButtons(ax_toolbar, labels, active=0)

        def on_class_selected(label):
            for c in ANNOTATION_ORDER:
                if CLASSES[c] == label:
                    current_class["idx"] = c
                    break

        radio.on_clicked(on_class_selected)

        # 点击 Classes 文字也可选中：检查是否点在某个 label 文本上
        def on_toolbar_click(event):
            if event.inaxes != ax_toolbar or event.button != 1:
                return
            for idx, text in enumerate(radio.labels):
                contains, _ = text.contains(event)
                if contains:
                    radio.set_active(idx)
                    on_class_selected(labels[idx])
                    fig.canvas.draw_idle()
                    break

        fig.canvas.mpl_connect("button_press_event", on_toolbar_click)

        # 把对象挂在 ax_toolbar 上，后续图片复用
        ax_toolbar._class_radio = radio
    else:
        radio = ax_toolbar._class_radio

    # Finish / preview / boundary / polygon / assist buttons live under the image area.
    # We create them per-image, but clean them up at the end of this function
    # to avoid stacking multiple overlapping widgets across images.
    extra_axes = []

    # Finish button: end annotation for this image and trigger save/post-process
    finish_ax = fig.add_axes([0.80, 0.045, 0.17, 0.035])
    extra_axes.append(finish_ax)
    finish_btn = mwidgets.Button(finish_ax, _ui_text("finish_image"))
    # 让语言切换时可以刷新按钮文本
    try:
        fig._finish_btn = finish_btn
    except Exception:
        pass

    def on_finish_clicked(event):
        finish_requested["value"] = True
        try:
            fig.canvas.stop_event_loop()
        except Exception:
            pass

    finish_btn.on_clicked(on_finish_clicked)

    # Toggle preview button: show/hide overlay of all existing masks
    preview_state = {"show": False}
    # Five buttons in a row, evenly spaced under the image area
    # Preview ~0.04, Boundary ~0.23, Polygon ~0.42, Model assist ~0.61, Finish ~0.80
    preview_ax = fig.add_axes([0.04, 0.045, 0.17, 0.035])
    extra_axes.append(preview_ax)
    preview_btn = mwidgets.Button(preview_ax, _ui_text("preview_show"))
    try:
        fig._preview_btn = preview_btn
        fig._preview_showing = False
    except Exception:
        pass

    def on_preview_clicked(event):
        preview_state["show"] = not preview_state["show"]
        preview_btn.label.set_text(
            _ui_text("preview_hide") if preview_state["show"] else _ui_text("preview_show")
        )
        try:
            fig._preview_showing = preview_state["show"]
        except Exception:
            pass
        # Immediately refresh the display to reflect the new preview state
        c = current_class["idx"]
        class_name = CLASSES[c]
        class_label = c + 1
        ax_display.clear()
        if preview_state["show"] and np.any(combined_mask > 0):
            # Preview all labeled classes: overlay combined_mask with per-class colors
            ax_display.imshow(image_rgb)
            h_img, w_img = combined_mask.shape[:2]
            overlay = np.zeros((h_img, w_img, 4), dtype=np.float32)
            max_label = int(combined_mask.max())
            for label in range(1, max_label + 1):
                if label > len(OVERLAY_COLORS):
                    break
                r, g, b, a = OVERLAY_COLORS[label - 1]
                sel = (combined_mask == label)
                if not np.any(sel):
                    continue
                overlay[sel, 0] = r
                overlay[sel, 1] = g
                overlay[sel, 2] = b
                overlay[sel, 3] = a
            ax_display.imshow(overlay)
            ax_display.set_title(os.path.basename(image_path))
            ax_display.axis("off")
        else:
            # Normal editing view: show only the current class mask (if any) on top of the raw image.
            class_mask = (combined_mask == class_label).astype(np.uint8)
            if np.any(class_mask > 0):
                overlay_mask(
                    ax_display,
                    image_rgb,
                    class_mask,
                    PREVIEW_OVERLAY_COLOR,
                    title=os.path.basename(image_path),
                    show_edge=True,
                )
            else:
                ax_display.imshow(image_rgb)
                ax_display.set_title(os.path.basename(image_path))
                ax_display.axis("off")
        _overlay_boundary(ax_display, boundary_mask)
        _update_status_line(fig, class_name, _ui_text("prompt_click"), _ui_text("status_hint"))
        fig.canvas.draw_idle()

    preview_btn.on_clicked(on_preview_clicked)

    # Boundary button: let user draw a manual separating line on this image
    boundary_state = {"requested": False}
    boundary_ax = fig.add_axes([0.23, 0.045, 0.17, 0.035])
    extra_axes.append(boundary_ax)
    boundary_btn = mwidgets.Button(boundary_ax, _ui_text("boundary_button"))

    try:
        fig._boundary_btn = boundary_btn
    except Exception:
        pass

    def on_boundary_clicked(event):
        boundary_state["requested"] = True
        try:
            fig.canvas.stop_event_loop()
        except Exception:
            pass

    boundary_btn.on_clicked(on_boundary_clicked)

    # Polygon selection button: multi-point polygon region for current class
    polygon_state = {"requested": False}
    polygon_ax = fig.add_axes([0.42, 0.045, 0.17, 0.035])
    extra_axes.append(polygon_ax)
    polygon_btn = mwidgets.Button(polygon_ax, _ui_text("polygon_button"))

    try:
        fig._polygon_btn = polygon_btn
    except Exception:
        pass

    def on_polygon_clicked(event):
        polygon_state["requested"] = True
        try:
            fig.canvas.stop_event_loop()
        except Exception:
            pass

    polygon_btn.on_clicked(on_polygon_clicked)

    # Model assist button lives near the bottom center
    assist_ax = None
    if ASSIST_MODEL_ENABLED:
        assist_ax = fig.add_axes([0.61, 0.045, 0.17, 0.035])
        extra_axes.append(assist_ax)

    try:
        # Main interaction loop: user can freely switch classes and add regions
        while True:
            if _window_closed or finish_requested["value"]:
                break

            c = current_class["idx"]
            class_name = CLASSES[c]
            class_label = c + 1

            ax_display.clear()
            if preview_state["show"] and np.any(combined_mask > 0):
                # Preview all labeled classes: overlay combined_mask with per-class colors
                ax_display.imshow(image_rgb)
                h_img, w_img = combined_mask.shape[:2]
                overlay = np.zeros((h_img, w_img, 4), dtype=np.float32)
                max_label = int(combined_mask.max())
                for label in range(1, max_label + 1):
                    if label > len(OVERLAY_COLORS):
                        break
                    r, g, b, a = OVERLAY_COLORS[label - 1]
                    sel = (combined_mask == label)
                    if not np.any(sel):
                        continue
                    overlay[sel, 0] = r
                    overlay[sel, 1] = g
                    overlay[sel, 2] = b
                    overlay[sel, 3] = a
                ax_display.imshow(overlay)
                ax_display.set_title(os.path.basename(image_path))
                ax_display.axis("off")
            else:
                # Normal editing view: show only the current class mask (if any) on top of the raw image.
                class_mask = (combined_mask == class_label).astype(np.uint8)
                if np.any(class_mask > 0):
                    overlay_mask(
                        ax_display,
                        image_rgb,
                        class_mask,
                        PREVIEW_OVERLAY_COLOR,
                        title=os.path.basename(image_path),
                        show_edge=True,
                    )
                else:
                    ax_display.imshow(image_rgb)
                    ax_display.set_title(os.path.basename(image_path))
                    ax_display.axis("off")
            _overlay_boundary(ax_display, boundary_mask)
            _update_status_line(fig, class_name, _ui_text("prompt_click"), _ui_text("status_hint"))
            fig.canvas.draw()

            boundary_state["requested"] = False
            polygon_state["requested"] = False
            points = get_click_points(ax_display, fig, (h, w), model_assist_ax=assist_ax)

            # If user clicked "Add boundary", enter boundary drawing mode
            if boundary_state["requested"]:
                boundary_mask = draw_boundary_line(ax_display, fig, boundary_mask)
                # After drawing, immediately refresh view and continue main loop
                continue

            # If user clicked Polygon region, enter polygon selection mode
            if polygon_state["requested"]:
                mask_binary = polygon_click_region(fig, ax_display, (h, w))
                polygon_state["requested"] = False
                if mask_binary is None or not np.any(mask_binary > 0):
                    continue

                # Apply manual boundary: cut the region along user-drawn lines
                if np.any(boundary_mask > 0):
                    mask_binary = mask_binary.copy()
                    mask_binary[boundary_mask > 0] = 0

                overlay_mask(
                    ax_display,
                    image_rgb,
                    mask_binary,
                    PREVIEW_OVERLAY_COLOR,
                    title=f"{class_name} — Accept this polygon region?",
                    show_edge=True,
                )
                _overlay_boundary(ax_display, boundary_mask)
                fig.canvas.draw()
                fig.canvas.flush_events()
                accept = wait_yes_no_in_window(
                    fig,
                    class_name,
                    "Accept this polygon region?  [ENTER]=Accept  [ESC]=Re-select",
                )
                if finish_requested["value"]:
                    break
                if accept is None:
                    return False
                if accept == "y":
                    c_now = current_class["idx"]
                    class_label_now = c_now + 1
                    combined_mask[mask_binary.astype(bool)] = class_label_now
                continue

            if finish_requested["value"]:
                break
            if points is None:
                return False
            if isinstance(points, tuple):
                mode = points[0]
                if mode == "model_assist":
                    # Model-assisted path: draw rectangle -> run segmentation model -> accept/reject
                    box = get_rectangle(fig, ax_display)
                    if finish_requested["value"]:
                        break
                    if box is None:
                        _update_status_line(fig, class_name, "Rectangle cancelled", "")
                        fig.canvas.draw()
                        plt.pause(0.3)
                        continue
                    loaded = load_assist_model()
                    if loaded is None:
                        _update_status_line(fig, class_name, "Assist model unavailable", "")
                        fig.canvas.draw()
                        plt.pause(0.8)
                        continue
                    model_assist, device_assist = loaded
                    try:
                        mask_binary = predict_assist_mask(
                            image_rgb, model_assist, device_assist, c, box_xyxy=box
                        )
                    except Exception as e:
                        _update_status_line(fig, class_name, f"Model error: {e}", "")
                        fig.canvas.draw()
                        plt.pause(1.0)
                        continue
                    # Apply manual boundary: cut the mask along user-drawn lines
                    if mask_binary is not None and np.any(boundary_mask > 0):
                        mask_binary = mask_binary.copy()
                        mask_binary[boundary_mask > 0] = 0

                    if mask_binary is None or not np.any(mask_binary > 0):
                        _update_status_line(
                            fig,
                            class_name,
                            "Model returned no pixels",
                            "",
                        )
                        ax_display.clear()
                        ax_display.imshow(image_rgb)
                        ax_display.set_title(f"{class_name} — No mask (model assist)")
                        ax_display.axis("off")
                        fig.canvas.draw()
                        wait_yes_no_in_window(
                            fig,
                            class_name,
                            "Model returned no pixels. Press ESC to skip.",
                        )
                        continue
                    overlay_mask(
                        ax_display,
                        image_rgb,
                        mask_binary,
                        PREVIEW_OVERLAY_COLOR,
                        title=f"{class_name} — Accept this mask? (model assist)",
                        show_edge=True,
                    )
                    _overlay_boundary(ax_display, boundary_mask)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.05)
                    accept = wait_yes_no_in_window(
                        fig,
                        class_name,
                        "Accept this mask?  [ENTER]=Accept  [ESC]=Re-select",
                    )
                    if finish_requested["value"]:
                        break
                    if accept is None:
                        return False
                    if accept == "y":
                        # Use the class that is currently selected in the toolbar at the time of acceptance
                        c_now = current_class["idx"]
                        class_label_now = c_now + 1
                        combined_mask[mask_binary.astype(bool)] = class_label_now
                    # Go back to main loop: user can keep annotating this class or switch classes
                    continue
                elif mode == "polygon":
                    # Freehand region path: use the enclosed area directly as a mask for the current class
                    stroke = points[1]
                    if stroke is None or len(stroke) < 3:
                        continue
                    import cv2

                    poly = np.array(stroke, dtype=np.float32)
                    # Clip to image bounds and convert to integer pixel indices
                    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
                    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
                    poly_int = poly.astype(np.int32)
                    mask_binary = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask_binary, [poly_int], 1)

                    # Apply manual boundary: cut the region along user-drawn lines
                    if np.any(boundary_mask > 0):
                        mask_binary[boundary_mask > 0] = 0

                    overlay_mask(
                        ax_display,
                        image_rgb,
                        mask_binary,
                        PREVIEW_OVERLAY_COLOR,
                        title=f"{class_name} — Accept this freehand region?",
                        show_edge=True,
                    )
                    _overlay_boundary(ax_display, boundary_mask)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    accept = wait_yes_no_in_window(
                        fig,
                        class_name,
                        "Accept this freehand region?  [ENTER]=Accept  [ESC]=Re-select",
                    )
                    if finish_requested["value"]:
                        break
                    if accept is None:
                        return False
                    if accept == "y":
                        c_now = current_class["idx"]
                        class_label_now = c_now + 1
                        combined_mask[mask_binary.astype(bool)] = class_label_now
                    continue

            if len(points) == 0:
                _update_status_line(fig, class_name, "No points selected, skipping", "")
                fig.canvas.draw()
                plt.pause(0.8)
                continue

            point_coords = np.array(points, dtype=np.float32)
            mask_binary = generate_mask(predictor, image_rgb, point_coords)
            if mask_binary is None:
                continue

            # Apply manual boundary: cut the region along user-drawn lines,
            # then re-run connected-component filtering so that only the
            # component(s) containing the user's click points are kept.
            if np.any(boundary_mask > 0):
                mask_binary = mask_binary.copy()
                mask_binary[boundary_mask > 0] = 0
                try:
                    mask_binary = _keep_only_components_containing_points(
                        mask_binary, point_coords
                    )
                except Exception:
                    # Fallback: if anything goes wrong, still use the cut mask
                    pass

            overlay_mask(
                ax_display,
                image_rgb,
                mask_binary,
                PREVIEW_OVERLAY_COLOR,
                title=f"{class_name} — Accept this mask?",
                show_edge=True,
            )
            _overlay_boundary(ax_display, boundary_mask)
            fig.canvas.draw()
            fig.canvas.flush_events()
            accept = wait_yes_no_in_window(
                fig,
                class_name,
                "Accept this mask?  [ENTER]=Accept  [ESC]=Re-select",
            )
            if finish_requested["value"]:
                break
            if accept is None:
                return False
            if accept == "n":
                continue

            # Use the class that is currently selected in the toolbar at the time of acceptance
            c_now = current_class["idx"]
            class_label_now = c_now + 1
            combined_mask[mask_binary.astype(bool)] = class_label_now
            # After accepting, immediately allow user to draw another region for this class,
            # or freely switch to another class via the toolbar.
            continue
    finally:
        # Remove per-image widget axes so they don't accumulate across images
        for ax_extra in extra_axes:
            try:
                ax_extra.remove()
            except Exception:
                pass

    if _window_closed:
        return False

    # 先按距离关系填充所有未标注像素（Voronoi 风格），保证最终没有 label=0 的“空洞”
    if FILL_GAPS_ENABLED:
        combined_mask = fill_gaps_in_combined_mask(
            combined_mask, len(CLASSES), FILL_GAPS_MAX_DISTANCE
        )

    # 再做一次按类别的最终后处理：去掉很小的毛刺、填小洞、用小圆盘做 closing 平滑边界
    # 这样平滑直接作用在最终要保存的多类别 mask 上，视觉效果更一致，
    # 同时不会破坏 FILL_GAPS_EXCLUDE_CLASSES 对器械等类别的约束。
    combined_mask = postprocess_combined_mask(combined_mask, len(CLASSES))

    # Save combined mask and overlay image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(MASKS_DIR, base_name + ".png")
    save_mask(combined_mask, save_path)
    mask_image_path = os.path.join(IMAGES_MASK_DIR, base_name + ".png")
    save_mask_image(image_rgb, combined_mask, mask_image_path)
    return True


def main():
    """List images, run annotation for each, handle window close."""
    global _window_closed, _settings_dirty

    # 先加载可视化可编辑的配置（classes 名称、模型路径等）
    _load_settings()

    if not os.path.isdir(IMAGES_DIR):
        print(f"Images folder not found: {IMAGES_DIR}")
        print("Creating it. Please add some images and run again.")
        os.makedirs(IMAGES_DIR, exist_ok=True)
        return

    predictor, device = load_model()
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(IMAGES_MASK_DIR, exist_ok=True)

    # Gather image paths (common extensions)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = []
    for f in sorted(os.listdir(IMAGES_DIR)):
        p = os.path.join(IMAGES_DIR, f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
            image_paths.append(p)

    # Skip images that already have a saved mask/overlay so we can resume where we left off
    pending_paths = []
    for p in image_paths:
        base_name = os.path.splitext(os.path.basename(p))[0]
        mask_path = os.path.join(MASKS_DIR, base_name + ".png")
        overlay_path = os.path.join(IMAGES_MASK_DIR, base_name + ".png")
        if os.path.exists(mask_path) or os.path.exists(overlay_path):
            print(f"Skipping already annotated image: {p}")
            continue
        pending_paths.append(p)
    image_paths = pending_paths

    if not image_paths:
        print(f"No images to annotate in {IMAGES_DIR} (all have masks already).")
        return

    print(f"Found {len(image_paths)} image(s).")

    def _create_main_figure():
        """Create main Matplotlib figure and axes with toolbar + settings button."""
        # 稍微放大默认窗口尺寸，适配 1080p 及以上分辨率，减少用户手动放大需求
        fig_local = plt.figure(figsize=(15, 9))
        gs_local = fig_local.add_gridspec(1, 2, width_ratios=[0.6, 3.4])
        ax_toolbar_local = fig_local.add_subplot(gs_local[0, 0])
        ax_local = fig_local.add_subplot(gs_local[0, 1])
        fig_local.subplots_adjust(bottom=0.10)
        fig_local._status_text = fig_local.text(
            0.5, 0.02, "", ha="center", fontsize=10, transform=fig_local.transFigure
        )
        fig_local.canvas.mpl_connect("close_event", _on_close)

        # 右上角 Settings 按钮
        try:
            settings_ax_local = fig_local.add_axes([0.84, 0.92, 0.12, 0.05])
            settings_btn_local = mwidgets.Button(settings_ax_local, _ui_text("settings_button"))

            def _on_settings_clicked(event):
                try:
                    open_settings_dialog(fig_local)
                except Exception as e:
                    print(f"Settings error: {e}")

            settings_btn_local.on_clicked(_on_settings_clicked)
            # 保存引用，语言切换时可刷新文字
            fig_local._settings_btn = settings_btn_local
            fig_local._ax_toolbar = ax_toolbar_local
            fig_local._ax_display = ax_local
        except Exception as e:
            print(f"Failed to create Settings button: {e}")

        _update_status_line(fig_local, "", _ui_text("main_initial"), "")
        plt.show(block=False)
        return fig_local, ax_local, ax_toolbar_local

    fig = None

    for image_path in image_paths:
        # 如果设置在运行期间被修改，进入下一张图前重载模型，
        # 并在下一张图重新创建主窗口，避免 Matplotlib 状态在多图之间“串台”
        if _settings_dirty:
            try:
                predictor, device = load_model()
            except SystemExit:
                raise
            except Exception as e:
                print(f"Failed to reload SAM model with new settings: {e}")
            _settings_dirty = False

        # 每一张图都使用一个全新的 figure / axes，彻底隔离所有交互状态和 widget。
        if fig is not None:
            plt.close(fig)
        fig, ax, ax_toolbar = _create_main_figure()

        _window_closed = False
        print(f"\n--- Image: {image_path} ---")

        img = Image.open(image_path).convert("RGB")
        image_rgb = np.array(img)

        success = annotate_image(predictor, device, image_path, image_rgb, fig, ax, ax_toolbar)
        if not success or _window_closed:
            # 用户点击右上角关闭窗口：直接终止整个进程，不再进入下一张图片
            print("Window closed. Exiting.")
            break

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
