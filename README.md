# ComfyUI_H3_VisQuasiDLSS5_Refiner

![ComfyUI](https://img.shields.io/badge/ComfyUI-✓-orange) ![MiniMax_H3](https://img.shields.io/badge/MiniMax%20H3-✓-blue) ![License](https://img.shields.io/badge/License-Apache--2.0-green)
## 完整说明

H3 Vis Quasi-DLSS5 Refiner 是一款面向 MiniMax H3 视频生成管线的修复增强节点。它借鉴 GPU 实时超分（如 DLSS）"以重生成代替直出"的思路，将二次采样技术落地为视频级画质修复：不追求像素级重建，而是让模型对目标区域做低强度潜空间重生成（re-generate）——以较低的 denoise 对画面进行二次采样，让模型在原有结构基础上"重新画一遍"，在回收伪影、抹除瑕疵的同时补回细节，实现近似超分 / 修复的观感提升。

因这种"低 denoise 重生成模拟超分增强"的运作逻辑与 DLSS 的"重建优于渲染"理念一脉相承，故命名为 Quasi-DLSS5——"类似 DLSS5"，而非真实的深度学习超分辨率。

核心流程：开放词汇目标检测（YOLO-World）→ Prompt 注入 → 分块潜空间重生成 → Mask 羽化回贴，整条链路自动完成，用户只需指定要修复的目标和强度。

YOLO-World 引导的 MiniMax H3 视频**局部修复（Refine）**节点 

检测 → Prompt 定义 → 分块潜空间重生成 → Mask 羽化回贴，整条管线收敛在**单个节点**内完成，不改动检测区域之外的任何像素。

## 特性

- **开放词汇目标检测**：YOLO-World（`yolov8s-world.pt` / `yolov8l-world.pt`），不限定类别，`detect_classes` 逗号分隔任意类名（face、hand、text、logo…）。
- **自动 Prompt 注入**：检测到的类名自动替换 `fix_prompt` 中的 `{classes}` 占位符，并通过 `detected_prompt` 输出口导出实际使用的完整提示词。
- **像素级保留**：Mask 羽化 + 时间平滑，只重生成检测区域；未检测区域逐像素保持原图（Paste-back 由 per-frame mask 加权）。
- **分块时空重生成**：帧数自动对齐 H3 网格（`_align_frame_count`，`n % 17 == 5`），长视频按时序分块（`chunk_frames`）逐块采样，控制显存峰值。
- **双模式**：
  - `full_frame_repair = true`：全帧去伪影（纹理闪烁/撕裂、色带、坏帧、色块、噪点等），跳过 YOLO。
  - `full_frame_repair = false`：盒级局部修复，YOLO 检测 → 类名注入修复 Prompt。
- **Headless RTX VSR**（可选）：内置 `nvvfx` 同倍率（1x，不放大）RTX Video Super Resolution 清理 + 时域 DC 稳定，失败自动降级放行，不影响 H3 修复结果。
- **检测加速**：`detect_step > 1` 时只对采样帧推理，中间帧复用最近一帧的检测框，配合时间 Mask 平滑容忍误差。
- **自包含**：检测、潜空间注入、条件构建、采样、回贴全部基于 ComfyUI 官方核心（`comfy_extras.nodes_minimax_h3` 等）实现，不依赖 Director / FaceRefine 插件包。
- 

## 安装

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：
   ```bash
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/YHZR9493/ComfyUI_H3_VisQuasiDLSS5_Refiner.git
   ```
2. 安装 Python 依赖：
   ```bash
   pip install ultralytics
   # 可选（RTX VSR 增强需要，RTX GPU + 官方 nvidia-vfx）：
   pip install nvvfx
   ```
3. 重启 ComfyUI。

节点位于分类 **`H3 Vis Quasi DLSS5`** 下。

## 所需模型

| 用途 | 模型 | 说明 |
|---|---|---|
| 主模型 | **MiniMax H3（ref2va）UNet** | 如 `minimax_h3_ref2va_int8_convrot.safetensors` 等，接入 `MODEL` |
| 视频 VAE | **MiniMax H3 Video VAE** | `minimax_h3_video_vae_fp16.safetensors`，接入 `video_vae` |
| 文本 | **MiniMax H3 CLIP** | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` 等，接入 `clip` |
| 检测 | **YOLO-World 权重** | `yolov8s-world.pt` / `yolov8l-world.pt`，本地路径填到 `yolov8_weights` |

> 注意：`yolov8_weights` 默认值为作者本机路径，使用前请改为你本机的权重绝对路径。

## 依赖的其他节点 / 运行库

该节点是**自包含**的，不依赖第三方节点包，仅依赖：

- **ComfyUI 官方核心模块**（随 ComfyUI 自带，无需额外安装）：
  - `comfy_extras.nodes_minimax_h3` — `MiniMaxH3ImageToVideo`（条件+空 AV latent 模板）、`MiniMaxH3SigmaShift`
  - `comfy_extras.nodes_custom_sampler` — `BasicGuider` / `CFGGuider` / `BasicScheduler` / `KSamplerSelect` / `RandomNoise` / `SamplerCustomAdvanced`
  - `comfy_extras.nodes_lt` — `LTXVSeparateAVLatent`
  - `nodes` — `VAEDecode`
  - `comfy.nested_tensor`、`comfy.utils`
- **PyTorch / ComfyUI 自带 torch**。
- **ultralytics**（YOLO-World，必需，懒加载）。
- **nvvfx**（可选，仅 `rtx_enhance` 开启时使用，失败自动跳过）。

## 使用方法

### 输入

| 接口 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `images` | IMAGE | ✔ | 输入视频帧序列 `[T,H,W,C]` |
| `model` | MODEL | ✔ | MiniMax H3 模型 |
| `video_vae` | VAE | ✔ | MiniMax H3 视频 VAE |
| `clip` | CLIP | ✔ | MiniMax H3 文本编码器 |

### 可选参数（节选）

| 参数 | 默认 | 说明 |
|---|---|---|
| `full_frame_repair` | true | true=全帧去伪影；false=YOLO 盒级局部修复 |
| `detect_classes` | `face, hand` | 盒级模式的目标类名（逗号分隔开放词汇） |
| `fix_prompt` | 去伪影 Prompt | 修复 Prompt，`{classes}` 会被替换 |
| `confidence` | 0.25 | YOLO 置信度阈值 |
| `box_dilation` / `mask_feather` | 12 / 6 | 检测框膨胀 / Mask 羽化半径 |
| `temporal_smooth` | 3 | Mask 时间平滑窗口 |
| `chunk_frames` | 124 | 分块采样帧数（自动对齐 H3 网格） |
| `denoise` / `steps` / `cfg` | 0.32 / 4 / 1.0 | 采样参数 |
| `rtx_enhance` / `rtx_quality` | true / ULTRA | 可选 headless RTX VSR 增强 |

### 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| `images` | IMAGE | 修复后的帧序列 |
| `detected_prompt` | STRING | 实际使用（已注入类名）的完整修复 Prompt |

## 兼容性

- 旧节点名 `H3YoloWorldDefineRefine` 仍可加载，行为与 `H3VisQuasiDLSS5Refiner` 完全一致。
- 需要 ComfyUI 已提供 MiniMax H3 官方节点支持（`comfy_extras.nodes_minimax_h3`）。

## License

Apache-2.0
