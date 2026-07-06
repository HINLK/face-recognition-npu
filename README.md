# 🧠 M系列芯片 NPU 人脸识别系统

> Face Recognition powered by Apple Neural Engine (ANE) on M-series chips

利用 Apple Silicon (M1/M2/M3/M4) 内置的 **Neural Engine (NPU)** 进行硬件加速的人脸检测与识别。

## ⚡ 加速原理

```
输入图像
  │
  ▼
┌──────────────────────────────────────┐
│  Vision Framework (NPU 自动调度)      │
│  · VNDetectFaceRectanglesRequest     │  ← 人脸检测
│  · VNDetectFaceLandmarksRequest      │  ← 关键点定位
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│  人脸对齐 (CPU, 轻量仿射变换)          │
│  · 旋转 → 双眼水平                    │
│  · 缩放 → 标准化尺寸                  │
│  · 输出 112×112 标准人脸              │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│  Core ML (computeUnits=ALL → NPU)    │
│  · MobileFaceNet 深度学习模型         │
│  · 输出 256 维嵌入向量                │
└──────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────┐
│  余弦相似度匹配                        │
│  · 对比数据库中所有向量               │
│  · 最高相似度 > 阈值 → 识别成功       │
└──────────────────────────────────────┘
```

| 阶段 | 框架/库 | 硬件 | 说明 |
|------|---------|------|------|
| 人脸检测 | `VNDetectFaceRectanglesRequest` | **Neural Engine** | 操作系统自动调度到 NPU |
| 关键点定位 | `VNDetectFaceLandmarksRequest` | **Neural Engine** | 65 个面部关键点 |
| 特征提取 | Core ML `computeUnits=ALL` | **Neural Engine** | 卷积/矩阵运算在 NPU 执行 |
| 人脸对齐 | OpenCV `warpAffine` | CPU | 轻量仿射变换 |

## 🔧 快速开始

```bash
# 第一步: 下载 AI 模型 (约 5-15MB)
python face_recognition_npu.py download-model

# 第二步: 注册人脸
python face_recognition_npu.py register --name "张三" --image photo.jpg

# 第三步: 开始识别
python face_recognition_npu.py recognize --image test.jpg --output result.jpg

# 实时摄像头
python face_recognition_npu.py webcam

# 查看已注册用户
python face_recognition_npu.py list

# 查看系统状态
python face_recognition_npu.py info
```

## 📋 依赖

脚本会**自动安装**所有缺失依赖，无需手动操作：

| 包名 | 用途 |
|------|------|
| `numpy` | 向量运算、矩阵计算 |
| `opencv-python` | 图像读写、预处理、显示 |
| `Pillow` | 图像处理（备用） |
| `pyobjc-framework-Vision` | 桥接 macOS Vision API → NPU 加速 |
| `pyobjc-framework-Quartz` | 桥接 Core Graphics 图形类型 |
| `coremltools` | 加载和运行 Core ML 模型 |

## 💻 系统要求

- **macOS 12.0+** (Monterey 或更新)
- **Apple Silicon Mac** (M1 / M2 / M3 / M4)
- 摄像头权限（用于 `webcam` 命令）

> ⚠️ Intel Mac 上可以运行，但无法使用 Neural Engine 加速，
> 性能会明显下降。

## 📁 数据存储

所有持久化数据存储在 `~/.face_recognition_npu/`：

```
~/.face_recognition_npu/
├── config.json                 # 运行时配置（阈值等）
├── face_database.pkl           # 已注册用户的人脸嵌入向量
├── MobileFaceNet.mlpackage/    # Core ML 人脸嵌入模型
├── deploy.prototxt             # OpenCV 备用检测模型
└── res10_300x300_ssd...model   # OpenCV 备用检测权重
```

## 📄 License

GNU General Public License v3.0 — 详见 [LICENSE](LICENSE)
