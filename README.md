# ComfyUI_H3_VisQuasiDLSS5_Refiner

![ComfyUI](https://img.shields.io/badge/ComfyUI-✓-orange) ![MiniMax_H3](https://img.shields.io/badge/MiniMax%20H3-✓-blue) ![License](https://img.shields.io/badge/License-Apache--2.0-green)

## 完整说明

H3 Vis Quasi-DLSS5 Refiner 是一款面向 MiniMax H3 视频生成管线的**无头超分增强节点**。它把 NVIDIA 超分重建（真 DLSS SR / RTX VSR）直接应用在输入画面上，按 `rtx_scale` 放大并重建细节，可选在放大帧上追加 DLSS 5 Neural Rendering 外观，再叠加可选的去雾（de-fog / de-haze）后期外观。

相比依赖 H3 二次采样的旧版（YOLO 目标检测 + 潜空间局部重生成 + Mask 回贴），当前版本**移除了全部检测 / 修复 / 回贴链路**，只保留"增强直通"一条路径：输入视频帧 → NVIDIA 超分桥（DLSS5 后端优先，nvvfx RTX VSR 兜底）→（可选）DLSS 5 NR →（可选）去雾，输出增强后的画面。

## 特性

- **无头超分增强**：`rtx_enhance` 开启后运行 NVIDIA RTX Video Super Resolution / DLSS SR（`rtx_scale` 1.0~2.0x），不可用时自动跳过，不影响主流程。
- **DLSS5 后端优先**：`rtx_backend=auto` 时存在真 DLSS SR（vsdlsssr.dll）则走 DLSS SR，否则退回 nvvfx RTX VSR；也可强制 `dlss5` / `nvvfx`。
- **DLSS 5 Neural Rendering**（可选）：`dlss_nr_mode` 在 SR 重建后的放大帧上施加 NR 外观（Off / Neutral / faithful / Realistic detail / Strong detail），`dlss_nr_intensity` 调节强度。NR 在独立进程运行，几乎不占 torch 显存。
- **深度 + 光流引导**：DLSS SR 使用 Depth Anything V2（`dlss_depth_model`）与 RAFT 光流（`dlss_motion_model`）做引导，权重首次使用时自动下载；`dlss_motion_scale=0.5` 半分辨率估光流提速约 4 倍。
- **两种输出策略**：
  - `rtx_keep_upscaled=true`（默认）：直接输出放大后的分辨率，细节最大化；
  - `rtx_keep_upscaled=false`：放大后缩回原分辨率，并把超分重建的细节残差按 `rtx_detail_strength` 回注（输出尺寸不变、下游更快）。
- **显存友好**：`rtx_unload_models`（默认开）在超分前卸载 H3/VAE 模型栈并清缓存，解决 8GB 显存 OOM 根因；深度/光流/NR 均支持帧批控制（`dlss_chunk_frames` / `dlss_nr_chunk_frames`），NR 超过 2M 像素自动限制批大小。
- **去雾外观**（可选）：`defog_enabled` 对画面做提亮 + 减辉光 + 锐化 + 提阴影的增强层，再乘性混合回画面，用于雾感 / 灰蒙画面。
- **自包含**：除 ComfyUI 官方核心外，仅依赖可选的 NVIDIA 运行时（缺失时自动降级放行）。

## 安装

1. 将本仓库克隆到 ComfyUI 的 `custom_nodes` 目录：
   ```bash
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/YHZR9493/ComfyUI_H3_VisQuasiDLSS5_Refiner.git
   ```
2. 安装可选 Python 依赖（DLSS5 后端需要，RTX GPU + 对应运行时）：
   ```bash
   pip install nvvfx
   ```
   DLSS5 桥（`dlss5_bridge.py`）会复用相邻的 `ComfyUI-DLSS5` 包及其隔离运行环境（vsdlssnr.dll / vsdlsssr.dll / bridge_runner.py），缺失时自动降级到 nvvfx 或跳过。
3. 重启 ComfyUI。

节点位于分类 **`H3 Vis Quasi DLSS5`** 下。

## 所需模型

| 用途 | 模型 | 说明 |
|---|---|---|
| 主模型 | **MiniMax H3（ref2va）UNet** | 如 `minimax_h3_ref2va_int8_convrot.safetensors` 等，接入 `MODEL` |
| 视频 VAE | **MiniMax H3 Video VAE** | `minimax_h3_video_vae_fp16.safetensors`，接入 `video_vae` |
| 文本 | **MiniMax H3 CLIP** | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` 等，接入 `clip` |
| 深度引导 | **Depth Anything V2** | `dlss_depth_model=Small/Base/Large`，首次使用自动下载 |
| 光流引导 | **RAFT** | `dlss_motion_model=Small/Large`，首次使用自动下载 |

## 依赖的其他节点 / 运行库

- **ComfyUI 官方核心模块**（随 ComfyUI 自带）：`comfy.utils`、`comfy.model_management` 等。
- **PyTorch / ComfyUI 自带 torch**。
- **nvvfx**（可选，仅 nvvfx 后端使用，失败自动跳过）。
- **ComfyUI-DLSS5 包**（可选，DLSS5 SR / NR 后端）：节点桥自动发现相邻 `custom_nodes/ComfyUI-DLSS5`，缺失时降级到 nvvfx 或跳过。

## 使用方法

### 输入

| 接口 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `images` | IMAGE | ✔ | 输入视频帧序列 `[T,H,W,C]` |
| `model` | MODEL | ✔ | MiniMax H3 模型（兼容旧工作流保留） |
| `video_vae` | VAE | ✔ | MiniMax H3 视频 VAE（兼容旧工作流保留） |
| `clip` | CLIP | ✔ | MiniMax H3 文本编码器（兼容旧工作流保留） |

### 可选参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `rtx_enhance` | true | 无头 RTX VSR / DLSS SR 超分增强总开关 |
| `rtx_unload_models` | true | 超分前卸载 H3 模型栈并清缓存（8GB 显存友好） |
| `rtx_quality` | ULTRA | RTX 超分质量档位 LOW/MEDIUM/HIGH/ULTRA |
| `rtx_scale` | 1.5 | 超分放大倍率（1.0~2.0） |
| `rtx_keep_upscaled` | true | true=直接输出放大分辨率；false=缩回原分辨率并回注细节 |
| `rtx_backend` | auto | 增强引擎：auto / dlss5 / nvvfx |
| `dlss_quality` | Quality | DLSS SR 质量预设（Quality/Balanced/Performance/Ultra Performance/Ultra Quality/DLAA） |
| `dlss_depth_model` | Small | 深度引导模型（Small/Base/Large） |
| `dlss_motion_model` | Small | 光流引导模型（Small/Large） |
| `dlss_chunk_frames` | 4 | 深度/光流引导推理帧批大小（1~32） |
| `dlss_motion_scale` | 0.5 | RAFT 光流分辨率比例（0.5 半分辨率约快 4 倍） |
| `dlss_nr_mode` | Off | DLSS 5 Neural Rendering 外观（Off/Neutral / faithful/Realistic detail/Strong detail） |
| `dlss_nr_intensity` | 1.0 | NR 效果强度倍率（0.0~1.5） |
| `dlss_nr_chunk_frames` | 4 | NR 进程帧批大小（1~16，超 2M 像素自动限 2） |
| `rtx_detail_strength` | 0.85 | 缩回原分辨率时 SR 细节回注强度（仅 keep_upscaled=false） |
| `defog_enabled` | false | 去雾后期外观开关 |
| `defog_strength` | 0.5 | 去雾强度（0.0~1.0） |

### 输出

| 输出 | 类型 | 说明 |
|---|---|---|
| `images` | IMAGE | 增强后的帧序列 |
| `detected_prompt` | STRING | 运行状态说明（"RTX/DLSS/defog enhanced" 或 "no enhancement applied"） |

## 兼容性

- 需要 NVIDIA RTX 显卡（RTX VSR / DLSS 运行时）。无 NVIDIA 运行时或缺失 DLL 时节点**失败自动放行**，原样返回输入，不中断工作流。
- `model` / `video_vae` / `clip` 仅保留以兼容旧工作流，当前版本不参与推理。

## 更新日志

### 2026-09-15 — 精简为纯超分增强节点（v2）

- **移除**：YOLO-World 目标检测、自动 Prompt 注入、分块潜空间重生成、Mask 羽化回贴、全帧去伪影等全部修复链路（旧辅助函数、常量一并删除，节点文件 1374 行 → 609 行）；
- **保留并强化**：headless RTX VSR / DLSS SR 超分增强 + 时域 DC 稳定 + 细节注入式缩回；
- **新增 DLSS5 桥**：`dlss5_bridge.py` 接入真 DLSS SR 与 DLSS 5 Neural Rendering（深度/光流引导、独立进程 NR）；
- **新增去雾外观**：`defog_enabled` / `defog_strength` 乘性混合去雾后期；
- 参数精简为 17 项，与 INPUT_TYPES / run 签名完全一致。

## License

Apache-2.0
