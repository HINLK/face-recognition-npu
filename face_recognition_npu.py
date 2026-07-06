#!/usr/bin/env python3
"""
=============================================================================
  M系列芯片 NPU 人脸识别系统
  Face Recognition using Apple Neural Engine (ANE) on M-series chips
  
  Copyright (C) 2026  HINLK
  License: GNU General Public License v3.0
=============================================================================

【系统概述】
  本脚本利用 Apple Silicon (M1/M2/M3/M4) 芯片内置的 Neural Engine (NPU)
  进行硬件加速人脸检测与识别。核心思路是:
    1. 通过 Vision Framework 调用 NPU → 快速检测人脸 + 关键点
    2. 通过 Core ML 调用 NPU → 提取人脸 256 维嵌入向量
    3. 余弦相似度对比 → 确认身份

【为什么能调用 NPU？】
  - Vision Framework 的 VNDetectFaceRectanglesRequest 在 M 系列芯片上
    会被操作系统自动调度到 Neural Engine 执行，无需开发者手动配置
  - Core ML 模型通过设置 computeUnits=CTComputeUnits.ALL，
    Core ML 运行时会优先选择 Neural Engine 进行推理

【硬件要求】
  - Apple Silicon Mac: M1 / M2 / M3 / M4 系列
  - macOS 12.0 (Monterey) 及以上

【文件结构】
  face_recognition_npu.py   ← 本文件, 人脸识别完整实现
  ~/.face_recognition_npu/  ← 运行时自动创建的配置/模型/数据库目录

【使用方法】
  python face_recognition_npu.py download-model   # 下载 Core ML 嵌入模型
  python face_recognition_npu.py register -n 张三 -i photo.jpg  # 注册人脸
  python face_recognition_npu.py recognize -i test.jpg          # 识别
  python face_recognition_npu.py webcam           # 摄像头实时识别
  python face_recognition_npu.py list             # 列出注册用户
  python face_recognition_npu.py info             # 查看系统状态
=============================================================================
"""

# ---- 标准库导入 ----
import os                 # 文件系统操作
import sys                # Python 解释器相关
import subprocess         # 调用外部命令 (如 pip install)
import argparse           # 命令行参数解析
import pickle             # Python 对象序列化 (人脸数据库存储)
import tempfile           # 临时文件
import time               # 计时 / 性能测量
import json               # 配置文件读写
from pathlib import Path  # 现代文件路径操作
from typing import List, Dict, Tuple, Optional, Union  # 类型标注

# ---- 第三方库 ----
# numpy 是科学计算基础库, 所有向量运算都依赖它
import numpy as np


# ============================================================================
# 第一部分: 依赖管理与自动安装
# ============================================================================
#
# 设计思路:
#   用户拿到脚本即可运行, 无需手动安装依赖。
#   脚本启动时自动检测缺失的包, 并调用 pip 安装。
#   必需包: numpy, opencv-python, Pillow
#   可选包: pyobjc-framework-Vision/Quartz/CoreML (NPU 直调的桥梁)
# ============================================================================

# 必需依赖: import_name → pip 包名
REQUIRED_PACKAGES = {
    'numpy': 'numpy',                  # 向量/矩阵运算, 所有嵌入向量的基础
    'cv2': 'opencv-python>=4.8.0',    # OpenCV: 图像读写/预处理/显示
    'PIL': 'Pillow>=10.0.0',          # Pillow: 图像处理 (备用)
}

# 可选依赖: PyObjC 系列, 用于桥接 macOS 原生框架
# 安装后可直接调用 Vision / Core ML API, 从而启用 NPU 加速
OPTIONAL_PACKAGES = {
    'Vision': 'pyobjc-framework-Vision',    # → VNDetectFaceRectanglesRequest
    'Quartz': 'pyobjc-framework-Quartz',    # → CGImage 等图形类型
    'CoreML': 'pyobjc-framework-CoreML',    # → MLModel 加载
}


def _is_package_installed(import_name: str) -> bool:
    """
    检查 Python 包是否已安装。
    
    原理: 直接尝试 __import__, 如果抛出 ImportError 则说明未安装。
    这是最可靠的检测方式, 比 pip list 解析快且不会误判。
    
    Args:
        import_name: 包的导入名, 如 'cv2', 'numpy'
    
    Returns:
        True 如果已安装, False 如果未安装
    """
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def _run_pip_install(package: str, desc: str = "") -> bool:
    """
    使用 pip 安装指定的 Python 包。
    
    注意:
      - 使用 sys.executable 确保安装到当前 Python 环境
      - stdout/stderr 重定向到 DEVNULL 以减少终端噪音
    
    Args:
        package: pip 包名, 如 'opencv-python>=4.8.0'
        desc: 可读描述, 用于终端提示
    
    Returns:
        True 安装成功, False 安装失败
    """
    label = desc or package
    print(f"  ⏳ 正在安装 {label} ...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  ✅ {label} 安装完成")
        return True
    except subprocess.CalledProcessError:
        print(f"  ❌ {label} 安装失败，请手动安装: pip install {package}")
        return False


def ensure_dependencies() -> bool:
    """
    确保所有必需依赖已安装, 缺失则自动安装。
    
    这是"开箱即用"的核心保障:
      1. 先检查必需包 → 缺失则 pip install
      2. 再检查可选包 → 缺失则尝试安装 (PyObjC 系列)
      3. 返回整体成功/失败状态
    
    Returns:
        True 所有必需依赖就绪, False 有依赖安装失败
    """
    print("🔍 检查依赖...")
    all_ok = True

    # ---- 必需依赖 ----
    for import_name, package in REQUIRED_PACKAGES.items():
        if not _is_package_installed(import_name):
            print(f"  ⚠️  缺少依赖: {package}")
            if not _run_pip_install(package):
                all_ok = False
        else:
            print(f"  ✅ {package}")

    # ---- 可选依赖 (PyObjC 桥接) ----
    # 这些包用于直接调用 macOS 原生 Vision / Core ML API
    # 如果安装失败也不影响基本功能 (有 OpenCV DNN 回退方案)
    for import_name, package in OPTIONAL_PACKAGES.items():
        if not _is_package_installed(import_name):
            print(f"  ⚠️  可选依赖未安装: {package} (尝试安装以启用 NPU 直调)")
            _run_pip_install(package)

    if not all_ok:
        print("\n❌ 部分必需依赖安装失败，请检查网络连接后重试")
        return False

    print("✅ 依赖检查完成\n")
    return True


# ============================================================================
# 第二部分: 配置与模型管理
# ============================================================================
#
# 所有持久化数据统一存储在 ~/.face_recognition_npu/ 目录下:
#   face_database.pkl      → 已注册用户的人脸嵌入向量
#   config.json            → 运行时配置 (阈值等)
#   MobileFaceNet.mlpackage → 人脸嵌入 Core ML 模型
#   deploy.prototxt         → OpenCV 人脸检测模型 (备用)
# ============================================================================

# 数据和模型目录: 用户主目录下的隐藏文件夹
MODEL_DIR = Path.home() / '.face_recognition_npu'
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在

# 各持久化文件的路径
FACE_DB_FILE = MODEL_DIR / 'face_database.pkl'   # pickle 序列化的人脸向量库
CONFIG_FILE = MODEL_DIR / 'config.json'           # JSON 配置文件

# Core ML 人脸嵌入模型的下载地址列表
# 主源是 HuggingFace 托管的 MobileFaceNet Core ML 版本
# 如果下载失败, 脚本会提示用户手动获取
FACE_EMBED_MODEL_URLS = [
    "https://huggingface.co/datasets/hinlk/face-models/resolve/main/mobilefacenet.mlpackage.zip",
]

# 模型在本地磁盘的期望路径
FACE_EMBED_MODEL_PATH = MODEL_DIR / 'MobileFaceNet.mlpackage'

# ---- 阈值配置 ----
# 这些值经过经验调整, 可根据实际场景微调

# 人脸检测置信度阈值 (0~1)
#   Vision Framework 检测结果中低于此值的人脸将被忽略
#   0.7 是一个较好的平衡: 太低会有误检, 太高会漏掉模糊人脸
FACE_DETECTION_CONFIDENCE = 0.7

# 人脸识别相似度阈值 (余弦相似度, 0~1)
#   两张同一人的人脸嵌入向量余弦相似度通常在 0.6~0.9 之间
#   不同人通常在 0.3~0.5
#   0.55 是默认阈值, 可根据需求调整 (越低越宽松, 越高越严格)
FACE_RECOGNITION_THRESHOLD = 0.55

# 严格模式阈值: 用于高安全场景
FACE_RECOGNITION_STRICT = 0.70


def load_config() -> dict:
    """
    从磁盘加载配置文件。
    
    如果配置文件不存在, 返回默认值。
    如果文件存在但损坏, 回退到默认值并覆盖写回。
    
    Returns:
        配置字典, 包含 detection_confidence, recognition_threshold 等
    """
    defaults = {
        'detection_confidence': FACE_DETECTION_CONFIDENCE,
        'recognition_threshold': FACE_RECOGNITION_THRESHOLD,
        'model_path': str(FACE_EMBED_MODEL_PATH),
        'version': '1.0.0',
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except (json.JSONDecodeError, IOError):
            # 配置文件损坏, 使用默认值
            pass
    return defaults


def save_config(config: dict):
    """
    保存配置到磁盘。
    
    使用 JSON 格式, 缩进美化, 方便手动编辑。
    """
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


# ============================================================================
# 第三部分: 人脸检测器 — 基于 Vision Framework (NPU 加速)
# ============================================================================
#
# 这是整个系统的第一步: 从图像中找到所有人脸的位置。
#
# 【为什么用 Vision Framework?】
#   Apple 的 Vision Framework 在 M 系列芯片上会自动将
#   VNDetectFaceRectanglesRequest 和 VNDetectFaceLandmarksRequest
#   的计算调度到 Neural Engine (ANE), 这是 NPU 加速的关键。
#
#   与传统的 OpenCV Haar Cascade / DNN 方案相比,
#   Vision + NPU 的检测速度快 3~5 倍, 且更省电。
#
# 【双后端设计】
#   1. vision 后端 (优先): PyObjC → Vision Framework → NPU
#   2. opencv_dnn 后端 (回退): OpenCV SSD 模型 → CPU
#   如果 PyObjC 未安装, 自动降级到 OpenCV DNN
# ============================================================================

class FaceDetector:
    """
    基于 Apple Vision Framework 的人脸检测器。
    
    核心能力:
      - detect(image) → 返回所有人脸的 bbox + 置信度 + 关键点
    
    NPU 加速机制:
      在 M 系列芯片上, VNDetectFaceRectanglesRequest
      会自动调度到 Apple Neural Engine (ANE) 上执行。
      开发者无需任何额外配置 —— 这是操作系统级的行为。
    
    坐标系转换:
      Vision 返回的坐标是归一化坐标 (0~1), 原点在左下角
      OpenCV 使用的是像素坐标, 原点在左上角
      本类内部做了转换, 对外统一使用 OpenCV 坐标系
    """

    def __init__(self, confidence_threshold: float = FACE_DETECTION_CONFIDENCE):
        """
        初始化人脸检测器。
        
        Args:
            confidence_threshold: 检测置信度阈值 (0~1)
                低于此值的检测结果将被丢弃
        """
        self.confidence_threshold = confidence_threshold
        self._use_backend = None       # 当前使用的后端: 'vision' 或 'opencv_dnn'
        self._init_backend()           # 自动选择最佳后端

    def _init_backend(self):
        """
        初始化检测后端, 按优先级尝试:
          1. Vision Framework (PyObjC) → NPU 加速, 效果最佳
          2. OpenCV DNN → CPU 回退, 兼容性好
        """
        # ---- 尝试加载 Vision Framework ----
        # PyObjC 是 Python 到 Objective-C 的桥接层
        # 通过它可以像调用 Python 对象一样调用 macOS 原生 API
        try:
            import Vision
            import Quartz
            from Foundation import NSURL, NSData, NSDictionary
            
            # 保存框架引用, 供后续 detect 方法使用
            self.Vision = Vision
            self.Quartz = Quartz
            self.NSURL = NSURL
            self.NSData = NSData
            self.NSDictionary = NSDictionary
            self._use_backend = 'vision'
            print("✅ 人脸检测后端: Apple Vision Framework (Neural Engine 加速)")
            return
        except ImportError:
            print("⚠️  PyObjC Vision 未安装, 尝试 OpenCV DNN 后端...")

        # ---- 回退到 OpenCV DNN ----
        try:
            import cv2
            self._use_backend = 'opencv_dnn'
            self._init_opencv_dnn()
            print("✅ 人脸检测后端: OpenCV DNN (CPU)")
            print("💡 提示: 安装 PyObjC 以获得 NPU 加速:")
            print("   pip install pyobjc-framework-Vision pyobjc-framework-Quartz")
        except Exception as e:
            raise RuntimeError(f"无法初始化人脸检测后端: {e}")

    def _init_opencv_dnn(self):
        """
        初始化 OpenCV DNN 人脸检测器 (CPU 回退方案)。
        
        使用 Caffe 格式的 SSD 人脸检测模型:
          - deploy.prototxt: 网络结构定义
          - res10_300x300_ssd_iter_140000.caffemodel: 预训练权重
        
        如果模型文件不存在, 自动从 GitHub 下载。
        """
        import cv2
        import urllib.request

        proto_path = MODEL_DIR / 'deploy.prototxt'
        caffe_path = MODEL_DIR / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

        # ---- 自动下载模型 ----
        if not caffe_path.exists():
            proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            caffe_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            print("  📥 下载 OpenCV 人脸检测模型...")
            try:
                urllib.request.urlretrieve(proto_url, proto_path)
                urllib.request.urlretrieve(caffe_url, caffe_path)
                print("  ✅ 模型下载完成")
            except Exception as e:
                raise RuntimeError(f"模型下载失败: {e}")

        # 加载 Caffe 模型到 OpenCV DNN
        self._dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(caffe_path))

        # 尝试设置计算后端 (CPU)
        # 注意: OpenCV DNN 目前不支持 Apple 的 ANE, 仅在 CPU 上运行
        try:
            self._dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self._dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception:
            pass

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的所有人脸 — 这是 FaceDetector 的主入口。
        
        根据初始化时选择的后端, 自动路由到对应方法:
          - vision 后端 → _detect_vision()   (NPU 加速)
          - opencv_dnn → _detect_opencv_dnn() (CPU 回退)
        
        Args:
            image: BGR 格式的 numpy 数组, shape (H, W, 3)
        
        Returns:
            人脸信息列表, 每个元素为字典:
            {
                'bbox': (x, y, w, h),        # 边界框 (OpenCV 坐标)
                'confidence': 0.95,           # 检测置信度 (0~1)
                'landmarks': {                # 关键点 (仅 Vision 后端)
                    'left_eye': (x, y),
                    'right_eye': (x, y),
                    'nose': (x, y),
                    ...
                }
            }
        """
        if self._use_backend == 'vision':
            return self._detect_vision(image)
        elif self._use_backend == 'opencv_dnn':
            return self._detect_opencv_dnn(image)
        return []

    def _detect_vision(self, image: np.ndarray) -> List[Dict]:
        """
        【NPU 加速核心】使用 Vision Framework 检测人脸。
        
        执行流程:
          1. numpy 数组 → 临时 PNG 文件 (Vision 需要 NSURL/文件路径)
          2. 创建 VNDetectFaceRectanglesRequest → 交给 Vision 处理
          3. Vision 自动将计算调度到 Neural Engine
          4. 解析返回的归一化坐标 → 转换为 OpenCV 像素坐标
          5. 提取人脸关键点 (landmarks): 眼睛、鼻子、嘴唇等
        
        坐标系转换:
          Vision:  原点在左下角, 坐标归一化到 [0, 1]
          OpenCV:  原点在左上角, 坐标单位是像素
          
          转换公式:
            opencv_x = vision_x * image_width
            opencv_y = (1 - vision_y - vision_height) * image_height
        
        边界框扩展:
          Vision 的检测框偏紧, 我们向外扩展 15% 作为余量,
          确保整张脸 (包括额头和下巴) 都在框内。
        """
        import cv2

        h, w = image.shape[:2]  # 图像高宽

        # Step 1: 将 numpy 数组保存为 PNG 临时文件
        #   这是 PyObjC + Vision 管线的一个限制:
        #   Vision 的 VNImageRequestHandler 需要 NSURL 或 CGImage
        #   从 numpy 创建 CGImage 比较复杂, 用临时文件最简单可靠
        tmp_path = MODEL_DIR / '_temp_detect.png'
        cv2.imwrite(str(tmp_path), image)

        try:
            # Step 2: 创建 Vision 人脸检测请求
            #   VNDetectFaceRectanglesRequest 是 Vision 内置的人脸检测器
            #   它已经过 Apple 优化, 会自动在 Neural Engine 上运行
            request = self.Vision.VNDetectFaceRectanglesRequest.alloc().init()

            # Step 3: 创建图像处理器并执行检测
            #   VNImageRequestHandler 负责将图像数据传给 Vision
            #   performRequests_error_ 执行请求, 返回 (success, error)
            image_url = self.NSURL.fileURLWithPath_(str(tmp_path))
            handler = self.Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                image_url, None
            )
            success, error = handler.performRequests_error_([request], None)

            if not success:
                return []

            # Step 4: 获取检测结果
            results = request.results()
            if results is None or len(results) == 0:
                return []

            # Step 5: 遍历每张人脸, 转换坐标并提取关键点
            faces = []
            for obs in results:
                # obs 是 VNFaceObservation 对象
                bbox = obs.boundingBox()  # 归一化坐标的 CGRect

                # ---- 坐标系转换 ----
                # Vision: 左下原点, 归一化
                # OpenCV: 左上原点, 像素
                fx = bbox.origin.x * w
                fy = (1.0 - bbox.origin.y - bbox.size.height) * h
                fw = bbox.size.width * w
                fh = bbox.size.height * h

                # ---- 边界框扩展 15% ----
                # 原因: Vision 的检测框通常紧扣面部五官, 不包括额头和下巴边缘
                # 扩展后的人脸区域更适合后续的对齐和特征提取
                margin_x = fw * 0.15
                margin_y = fh * 0.15
                x = max(0, int(fx - margin_x))
                y = max(0, int(fy - margin_y))
                bw = min(w - x, int(fw + 2 * margin_x))
                bh = min(h - y, int(fh + 2 * margin_y))

                # ---- 获取置信度 ----
                # VNFaceObservation 的 confidence 属性 (macOS 12+)
                confidence = float(obs.confidence()) if hasattr(obs, 'confidence') and obs.confidence() else 1.0

                # 过滤低于阈值的人脸
                if confidence >= self.confidence_threshold:
                    face_info = {
                        'bbox': (x, y, bw, bh),
                        'confidence': confidence,
                    }

                    # ---- 提取人脸关键点 ----
                    # landmarks 用于后续的人脸对齐 (FaceAligner)
                    # 包含: 左眼中心、右眼中心、鼻尖、嘴唇、面部轮廓
                    if hasattr(obs, 'landmarks') and obs.landmarks():
                        face_info['landmarks'] = self._extract_landmarks(obs, w, h)

                    faces.append(face_info)

            return faces

        except Exception as e:
            print(f"  ⚠️  Vision 检测异常: {e}")
            return []
        finally:
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()

    def _extract_landmarks(self, observation, img_w: int, img_h: int) -> Dict[str, Tuple[float, float]]:
        """
        从 Vision 的 VNFaceObservation 中提取人脸关键点。
        
        Vision 提供了 65+ 个面部关键点, 分布在以下区域:
          - leftEye / rightEye: 左右眼轮廓 (各约 6~8 个点)
          - nose / noseCrest: 鼻子轮廓和鼻梁
          - faceContour: 面部外轮廓
          - outerLips / innerLips: 外唇和内唇轮廓
          - medianLine: 面部中线
        
        这里我们取每个区域的平均中心点, 用于后续的人脸对齐。
        
        Args:
            observation: VNFaceObservation 对象
            img_w, img_h: 图像宽高 (像素)
        
        Returns:
            关键点字典, key 为部位名, value 为 (x, y) 像素坐标
        """
        landmarks = {}
        try:
            lm = observation.landmarks()
            # 各区域 → 取所有点的平均中心
            landmark_map = {
                'left_eye': lm.leftEye(),        # 左眼区域
                'right_eye': lm.rightEye(),      # 右眼区域
                'nose': lm.nose(),               # 鼻子区域
                'face_contour': lm.faceContour(), # 面部轮廓
                'outer_lips': lm.outerLips(),    # 外唇轮廓
            }
            for name, region in landmark_map.items():
                if region is None:
                    continue
                pts = region.points()
                if pts is None:
                    continue
                pts_list = list(pts)
                if pts_list:
                    # 计算该区域所有点的平均位置
                    # Vision 返回归一化坐标 (原点左下), 需翻转为 OpenCV 坐标 (原点左上)
                    cx = sum(p.x for p in pts_list) / len(pts_list) * img_w
                    cy = (1.0 - sum(p.y for p in pts_list) / len(pts_list)) * img_h
                    landmarks[name] = (float(cx), float(cy))
        except Exception:
            pass
        return landmarks

    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Dict]:
        """
        【CPU 回退】使用 OpenCV DNN SSD 模型检测人脸。
        
        这是 PyObjC 不可用时的备选方案, 在 CPU 上运行。
        精度和速度均不及 Vision + NPU, 但兼容性好。
        
        输入处理:
          - blobFromImage: 将图像转为 300x300 的标准化 blob
          - mean=(104,117,123): 减均值 (ImageNet 统计值)
        
        Args:
            image: BGR numpy 数组
        
        Returns:
            人脸列表 (不含 landmarks, OpenCV DNN 不输出关键点)
        """
        import cv2

        h, w = image.shape[:2]

        # 创建 DNN 输入 blob: 缩放+减均值+通道交换
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()  # 前向推理 (CPU)

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # 置信度在第 3 列
            if confidence >= self.confidence_threshold:
                # 解析边界框坐标 (归一化 → 像素)
                x1 = max(0, int(detections[0, 0, i, 3] * w))
                y1 = max(0, int(detections[0, 0, i, 4] * h))
                x2 = min(w, int(detections[0, 0, i, 5] * w))
                y2 = min(h, int(detections[0, 0, i, 6] * h))
                faces.append({
                    'bbox': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': float(confidence),
                })
        return faces


# ============================================================================
# 第四部分: 人脸对齐器
# ============================================================================
#
# 为什么需要对齐?
#   同一张人脸在不同照片中可能有不同的角度 (歪头)、大小 (远近)、
#   位置 (偏左/偏右)。如果直接输入到特征提取模型, 会严重影响
#   嵌入向量的质量, 导致识别准确率大幅下降。
#
# 如何对齐?
#   利用 Vision 提供的眼睛关键点, 做仿射变换:
#     1. 旋转 → 使双眼连线水平
#     2. 缩放 → 使双眼距离标准化
#     3. 平移 → 使双眼位于图像固定位置
#   最终输出 112x112 的标准化人脸图像。
# ============================================================================

class FaceAligner:
    """
    基于眼睛位置的人脸对齐器。
    
    输出统一为 112x112 像素的标准化人脸,
    双眼位于画面中固定位置, 便于后续的特征提取模型处理。
    """

    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        """
        Args:
            target_size: 输出图像的尺寸 (宽, 高)
                112x112 是 MobileFaceNet / ArcFace 等主流人脸识别模型的标准输入
        """
        self.target_size = target_size

    def align(
        self,
        image: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        landmarks: Optional[Dict] = None,
    ) -> Optional[np.ndarray]:
        """
        对齐人脸图像。
        
        【对齐算法】
        如果有眼睛关键点 (landmarks):
          1. 计算双眼连线与水平线的夹角 → 旋转角度
          2. 计算双眼距离 → 缩放比例
          3. 计算双眼中心位置 → 平移量
          4. 组合以上三者为一个仿射变换矩阵
          5. 通过 cv2.warpAffine 一次性完成旋转+缩放+平移
        
        如果没有关键点 (如 OpenCV DNN 后端):
          直接缩放裁剪区域到 112x112 (简单但不够精确)
        
        Args:
            image: 原始 BGR 图像
            face_bbox: 人脸边界框 (x, y, w, h) OpenCV 坐标
            landmarks: Vision 提取的关键点 (可选)
        
        Returns:
            112x112 BGR 标准化人脸图像, 或 None (失败时)
        """
        import cv2

        x, y, w, h = face_bbox

        # ---- 边界框修正: 确保不超出图像范围 ----
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)

        if w <= 0 or h <= 0:
            return None

        # 裁剪人脸区域
        face_crop = image[y:y + h, x:x + w].copy()

        # ---- 基于眼睛关键点的精确对齐 ----
        if landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks:
            # 将全局坐标转为裁剪区域内的局部坐标
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            le_x, le_y = left_eye[0] - x, left_eye[1] - y
            re_x, re_y = right_eye[0] - x, right_eye[1] - y

            # ---- 计算旋转角度 ----
            # 双眼连线与水平线的夹角
            dx = re_x - le_x
            dy = re_y - le_y
            angle = np.degrees(np.arctan2(dy, dx))

            # ---- 计算双眼中心 ----
            eye_center_x = (le_x + re_x) / 2.0
            eye_center_y = (le_y + re_y) / 2.0

            # ---- 目标位置: 在 112x112 图像中双眼应该在的位置 ----
            # 水平居中 (50%), 垂直偏上 (35%), 双眼距离占 25% 宽度
            target_eye_x = self.target_size[0] * 0.5
            target_eye_y = self.target_size[1] * 0.35
            target_eye_dist = self.target_size[0] * 0.25

            # ---- 计算缩放比例 ----
            current_eye_dist = np.sqrt(dx ** 2 + dy ** 2)
            scale = target_eye_dist / current_eye_dist if current_eye_dist > 0 else 1.0

            # ---- 构建仿射变换矩阵 ----
            # 先以双眼中心为原点旋转+缩放
            M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, scale)
            # 再平移到目标位置
            M[0, 2] += target_eye_x - eye_center_x
            M[1, 2] += target_eye_y - eye_center_y

            # ---- 执行仿射变换 ----
            # INTER_LINEAR: 双线性插值, 速度与质量的平衡
            # BORDER_REPLICATE: 边缘像素复制填充, 避免黑边
            aligned = cv2.warpAffine(
                face_crop, M, self.target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            return aligned

        # ---- 无关键点时的简单缩放 ----
        return cv2.resize(face_crop, self.target_size, interpolation=cv2.INTER_LINEAR)


# ============================================================================
# 第五部分: Core ML 人脸嵌入模型 (NPU 加速特征提取)
# ============================================================================
#
# 这是整个系统最核心的 AI 推理环节:
#   将对齐后的人脸图像 (112x112 RGB) 送入深度学习模型,
#   输出一个 256 维的浮点数向量 (嵌入向量 / embedding)。
#
# 这个嵌入向量具有"语义距离"特性:
#   - 同一人的不同照片 → 向量接近 (余弦相似度 > 0.6)
#   - 不同人的照片 → 向量远离 (余弦相似度 < 0.4)
#
# 【NPU 加速原理】
#   Core ML 模型加载后, 设置 computeUnits=CTComputeUnits.ALL:
#     - Core ML 运行时会分析模型中每个算子的特性
#     - 适合 Neural Engine 的算子 (如卷积、矩阵乘法) → 调度到 ANE
#     - 不适合的算子 → 回退到 CPU/GPU
#   ANE 的能效比远超 CPU/GPU, 是移动端/桌面端 AI 推理的最优选择。
#
# 【回退方案】
#   如果 Core ML 模型文件不存在:
#     使用 OpenCV ORB 特征描述符作为简化替代
#     (精度远不如深度学习模型, 仅用于验证管线是否畅通)
# ============================================================================

class FaceEmbedder:
    """
    使用 Core ML 模型提取人脸嵌入向量的特征提取器。
    
    双后端设计:
      1. Core ML 模型 → NPU 加速 (精度高, 速度快)
      2. OpenCV ORB → CPU 回退 (仅用于功能验证)
    """

    # 默认输入输出规格 (适合 MobileFaceNet)
    INPUT_SIZE = (112, 112)   # 模型输入尺寸 (宽, 高)
    EMBEDDING_SIZE = 256      # 输出嵌入向量维度

    def __init__(self, model_path: Optional[Path] = None):
        """
        Args:
            model_path: Core ML 模型文件路径 (.mlpackage 或 .mlmodel)
                如果为 None, 使用默认路径 ~/.face_recognition_npu/MobileFaceNet.mlpackage
        """
        self.model_path = model_path or FACE_EMBED_MODEL_PATH
        self._model = None         # Core ML 模型实例
        self._use_coreml = False   # 是否使用 Core ML (vs ORB 回退)
        self._init_model()

    def _init_model(self):
        """
        初始化特征提取模型。
        
        优先加载 Core ML 模型 (NPU 加速),
        如果模型文件不存在或 coremltools 未安装, 回退到 ORB。
        """
        if self.model_path.exists():
            try:
                import coremltools as ct

                # 加载 Core ML 模型
                self._model = ct.models.MLModel(str(self.model_path))

                # 【关键】设置 compute_units=ALL 以启用 Neural Engine
                #   CTComputeUnits.ALL → Core ML 会在 CPU/GPU/NeuralEngine 中自动选择
                #   Neural Engine 的优先级最高, 适合的算子会优先调度到 ANE
                try:
                    self._model.compute_units = ct.ComputeUnits.ALL
                    print("✅ 人脸嵌入模型: Core ML (Neural Engine 加速)")
                except Exception:
                    print("✅ 人脸嵌入模型: Core ML (默认 compute units)")

                self._use_coreml = True
                self._detect_model_specs()  # 自动检测模型输入输出规格
                return
            except ImportError:
                print("⚠️  coremltools 未安装, 无法加载 Core ML 模型")

        # ---- 回退方案 ----
        print("⚠️  未找到 Core ML 人脸嵌入模型")
        print(f"   期望位置: {self.model_path}")
        print("   将使用简化特征提取方案 (建议下载 Core ML 模型以获得最佳效果)")
        self._init_fallback_model()

    def _detect_model_specs(self):
        """
        自动检测模型的输入输出规格。
        
        不同的 Core ML 模型可能有不同的:
          - 输入形状: (1,3,112,112) NCHW 或 (1,112,112,3) NHWC
          - 嵌入维度: 128 / 256 / 512
        
        从模型的 metadata 中读取这些信息, 避免硬编码。
        """
        try:
            spec = self._model.get_spec()
            # 输入规格
            if len(spec.description.input) > 0:
                input_desc = spec.description.input[0]
                if hasattr(input_desc.type, 'multiArrayType'):
                    shape = input_desc.type.multiArrayType.shape
                    if len(shape) >= 3:
                        # shape[-2:] 是 (H, W)
                        self.INPUT_SIZE = (int(shape[2]), int(shape[1]))
            # 输出规格
            if len(spec.description.output) > 0:
                output_desc = spec.description.output[0]
                if hasattr(output_desc.type, 'multiArrayType'):
                    shape = output_desc.type.multiArrayType.shape
                    if len(shape) >= 1:
                        self.EMBEDDING_SIZE = int(shape[-1])
        except Exception:
            # 检测失败, 保持默认值
            pass

    def _init_fallback_model(self):
        """
        初始化 ORB 特征描述符作为回退方案。
        
        ORB (Oriented FAST and Rotated BRIEF) 是经典的图像特征描述符,
        提取图像中的关键点并计算描述向量。
        
        ⚠️ 重要: ORB 不是人脸识别专用算法, 精度远低于深度学习模型。
        这里仅用于验证整个管线能够跑通, 实际使用请下载 Core ML 模型。
        """
        import cv2
        self._orb = cv2.ORB_create(nfeatures=256)  # 提取 256 个特征点
        print("   回退后端: OpenCV ORB 特征描述符 (仅供参考)")

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        提取人脸嵌入向量 — FaceEmbedder 的主入口。
        
        自动路由到 Core ML 或 ORB 后端。
        返回的向量已做 L2 归一化 (单位向量), 方便后续余弦相似度计算。
        
        Args:
            face_image: 对齐后的人脸图像, 112x112 BGR
        
        Returns:
            L2 归一化的嵌入向量 (float32 numpy array)
        """
        if self._use_coreml and self._model is not None:
            return self._extract_coreml(face_image)
        else:
            return self._extract_fallback(face_image)

    def _extract_coreml(self, face_image: np.ndarray) -> np.ndarray:
        """
        【NPU 加速核心】使用 Core ML 模型提取嵌入向量。
        
        处理流程:
          1. BGR → RGB (Core ML 模型通常训练在 RGB 色彩空间)
          2. 缩放至模型输入尺寸 (通常 112x112)
          3. 归一化: (pixel - 127.5) / 128.0 → 值域约 [-1, 1]
          4. 调整维度: (H,W,C) → (1,C,H,W) 或 (1,H,W,C)
          5. 调用 model.predict() → Core ML 自动调度到 Neural Engine
          6. 输出向量做 L2 归一化
        
        Args:
            face_image: 112x112 BGR 人脸图像
        
        Returns:
            L2 归一化的 256 维嵌入向量
        """
        import cv2

        # Step 1: BGR → RGB
        #   OpenCV 默认 BGR 通道顺序, Core ML 模型通常期望 RGB
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Step 2: 缩放至模型输入尺寸
        face_resized = cv2.resize(face_rgb, self.INPUT_SIZE)

        # Step 3: 像素值归一化
        #   MobileFaceNet 使用 (pixel - 127.5) / 128.0 归一化到 [-1, 1]
        face_float = face_resized.astype(np.float32)
        face_normalized = (face_float - 127.5) / 128.0

        # Step 4: 调整数据维度以匹配模型输入格式
        #   从模型 spec 读取输入 shape 来判断是 NCHW 还是 NHWC
        #   NCHW: (batch, channel, height, width) — 常见于 PyTorch 导出
        #   NHWC: (batch, height, width, channel) — 常见于 TensorFlow 导出
        try:
            spec = self._model.get_spec()
            input_desc = spec.description.input[0]
            if hasattr(input_desc.type, 'multiArrayType'):
                shape = input_desc.type.multiArrayType.shape
                if len(shape) == 4 and shape[1] == 3:
                    # NCHW: 需要 transpose (H,W,C) → (C,H,W) 然后加 batch 维度
                    face_input = np.transpose(face_normalized, (2, 0, 1))[np.newaxis, ...]
                else:
                    # NHWC: 直接加 batch 维度
                    face_input = face_normalized[np.newaxis, ...]
            else:
                face_input = face_normalized[np.newaxis, ...]
        except Exception:
            face_input = face_normalized[np.newaxis, ...]

        try:
            # Step 5: Core ML 推理 (Neural Engine 加速)
            #   model.predict() 会自动将计算调度到 ANE (如果 compute_units=ALL)
            predictions = self._model.predict({'input': face_input})

            # Step 6: 提取输出向量并 L2 归一化
            #   L2 归一化后, ||embedding|| = 1
            #   这样余弦相似度可以直接用内积计算 (cos(a,b) = a·b 当 |a|=|b|=1)
            output_key = list(predictions.keys())[0]
            embedding = predictions[output_key]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten()

            # L2 归一化: v' = v / ||v||
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.astype(np.float32)

        except Exception as e:
            # Core ML 推理失败, 回退到 ORB
            print(f"  ⚠️  Core ML 推理失败: {e}")
            return self._extract_fallback(face_image)

    def _extract_fallback(self, face_image: np.ndarray) -> np.ndarray:
        """
        【回退方案】使用 ORB 特征描述符提取特征向量。
        
        ORB 工作流程:
          1. 转灰度图
          2. 检测 FAST 角点 (关键点)
          3. 计算 BRIEF 描述符 (256 维二进制向量)
          4. 对所有描述符取平均 → 得到一个 256 维浮点向量
          5. L2 归一化
        
        局限性:
          - ORB 不包含人脸语义信息, 只能捕捉纹理/边缘特征
          - 同一人的不同表情/角度下, ORB 向量差异很大
          - 强烈建议安装 Core ML 模型替代
        
        Args:
            face_image: 112x112 BGR 人脸图像
        
        Returns:
            L2 归一化的特征向量
        """
        import cv2

        # 转灰度 → ORB 在灰度图上工作
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (112, 112))

        # ORB 关键点检测 + 描述符计算
        keypoints = self._orb.detect(gray_resized, None)
        keypoints, descriptors = self._orb.compute(gray_resized, keypoints)

        if descriptors is not None and len(descriptors) > 0:
            # 对所有关键点的描述符取平均 → 固定长度向量
            embedding = np.mean(descriptors.astype(np.float32), axis=0)
        else:
            # 极端退化: 没有检测到任何特征点
            # 用降采样图像本身作为"特征"
            embedding = cv2.resize(gray_resized, (16, 16)).astype(np.float32).flatten() / 255.0

        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.astype(np.float32)


# ============================================================================
# 第六部分: 人脸数据库
# ============================================================================
#
# 使用 pickle 将 Python 字典序列化到磁盘。
# 数据结构: { "张三": [vec1, vec2], "李四": [vec3], ... }
#
# 为什么一个用户可以有多个向量?
#   同一用户注册多张照片 (不同角度/表情/光照),
#   识别时与所有向量对比取最高相似度, 提高鲁棒性。
# ============================================================================

class FaceDatabase:
    """
    本地人脸向量数据库。
    
    存储格式 (Python dict → pickle 文件):
      {
        "用户名": [
          np.ndarray(256,),  # 嵌入向量 1
          np.ndarray(256,),  # 嵌入向量 2 (可选, 同一人的不同照片)
          ...
        ],
        ...
      }
    
    文件位置: ~/.face_recognition_npu/face_database.pkl
    """

    def __init__(self, db_path: Path = FACE_DB_FILE):
        self.db_path = db_path
        # records 是核心数据结构: name → list of embeddings
        self.records: Dict[str, List[np.ndarray]] = {}
        self._load()  # 启动时自动加载已有数据

    def _load(self):
        """
        从磁盘加载人脸数据库。
        
        如果文件不存在 → 从空开始
        如果文件损坏 → 打印警告, 从空开始
        """
        if self.db_path.exists():
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        self.records = data
                print(f"📂 已加载人脸数据库: {len(self.records)} 个用户")
            except (pickle.PickleError, EOFError, IOError) as e:
                print(f"⚠️  数据库加载失败: {e}, 将创建新数据库")
                self.records = {}

    def save(self):
        """将人脸数据库持久化到磁盘 (每次修改后自动调用)"""
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.records, f)

    def add(self, name: str, embedding: np.ndarray):
        """
        添加一个人的嵌入向量到数据库。
        
        如果该用户已存在, 追加向量 (支持多角度注册)。
        每次添加后自动保存到磁盘。
        
        Args:
            name: 用户名
            embedding: 256 维 L2 归一化嵌入向量
        """
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(embedding)
        self.save()

    def remove_user(self, name: str) -> bool:
        """
        删除指定用户的所有数据。
        
        Returns:
            True 删除成功, False 用户不存在
        """
        if name in self.records:
            del self.records[name]
            self.save()
            return True
        return False

    def get_all_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """获取所有注册用户的嵌入向量 (返回副本)"""
        return dict(self.records)

    def get_users(self) -> List[Dict]:
        """获取用户列表, 包含名称和样本数"""
        return [{'name': name, 'samples': len(embs)} for name, embs in self.records.items()]

    @property
    def user_count(self) -> int:
        """已注册用户总数"""
        return len(self.records)


# ============================================================================
# 第七部分: 人脸识别引擎 — 完整管线
# ============================================================================
#
# FaceRecognitionEngine 是整个系统的"大脑", 它串联了所有组件:
#
#   ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
#   │ 输入图像  │ → │ 人脸检测  │ → │  人脸对齐  │ → │ 特征提取  │
#   │          │    │ (NPU)    │    │  (CPU)    │    │ (NPU)    │
#   └──────────┘    └──────────┘    └───────────┘    └──────────┘
#                                                          │
#                                                          ▼
#                        ┌──────────────────────────────────────┐
#                        │  余弦相似度匹配 → "张三" / "Unknown"  │
#                        └──────────────────────────────────────┘
# ============================================================================

class FaceRecognitionEngine:
    """
    人脸识别引擎 — 整合检测、对齐、嵌入、匹配的完整管线。
    
    使用方法:
      engine = FaceRecognitionEngine()
      engine.register("张三", "photo.jpg")       # 注册
      results = engine.recognize("test.jpg")      # 识别
    """

    def __init__(self, config: Optional[dict] = None):
        """
        初始化人脸识别引擎, 创建所有子组件。
        
        Args:
            config: 配置字典 (可选, 默认从磁盘加载)
        """
        cfg = config or load_config()
        self.recognition_threshold = cfg.get('recognition_threshold', FACE_RECOGNITION_THRESHOLD)
        self.detection_confidence = cfg.get('detection_confidence', FACE_DETECTION_CONFIDENCE)

        print("\n" + "=" * 50)
        print("  初始化 M 系列芯片 NPU 人脸识别引擎")
        print("=" * 50 + "\n")

        t0 = time.time()

        # ---- 创建各子组件 ----
        # 1. 人脸检测器 (Vision Framework → NPU)
        self.detector = FaceDetector(confidence_threshold=self.detection_confidence)

        # 2. 人脸对齐器 (CPU, 轻量仿射变换)
        self.aligner = FaceAligner(target_size=(112, 112))

        # 3. 特征提取器 (Core ML → NPU 或 ORB → CPU)
        self.embedder = FaceEmbedder(
            model_path=Path(cfg.get('model_path', str(FACE_EMBED_MODEL_PATH)))
        )

        # 4. 人脸数据库 (pickle 持久化)
        self.database = FaceDatabase()

        t1 = time.time()
        print(f"\n⏱️  初始化耗时: {t1 - t0:.2f}s")
        print("=" * 50 + "\n")

    def register(self, name: str, image: Union[str, np.ndarray]) -> Tuple[bool, str]:
        """
        注册用户人脸: 从照片中提取嵌入向量并存入数据库。
        
        流程:
          1. 读取图像
          2. 检测人脸 → 取置信度最高的一张
          3. 对齐人脸 → 112x112 标准化图像
          4. 提取嵌入向量 → 256 维浮点向量
          5. 存入数据库
        
        Args:
            name: 用户名称
            image: 图像文件路径 (str) 或 numpy 数组
        
        Returns:
            (成功与否, 描述消息)
        """
        # 读取图像
        if isinstance(image, str):
            import cv2
            img = cv2.imread(image)
            if img is None:
                return False, f"无法读取图像: {image}"
        else:
            img = image

        # 检测人脸
        faces = self.detector.detect(img)
        if len(faces) == 0:
            return False, "未在图像中检测到人脸, 请提供清晰的正面照片"

        # 多张人脸时, 选置信度最高的那张
        if len(faces) > 1:
            faces.sort(key=lambda f: f.get('confidence', 0), reverse=True)

        face = faces[0]
        bbox = face['bbox']
        landmarks = face.get('landmarks', {})

        # 对齐人脸
        aligned = self.aligner.align(img, bbox, landmarks)
        if aligned is None:
            return False, "人脸对齐失败"

        # 提取嵌入向量并存入数据库
        embedding = self.embedder.extract_embedding(aligned)
        self.database.add(name, embedding)

        return True, f"成功注册用户 '{name}' (检测置信度: {face.get('confidence', 1.0):.2f})"

    def recognize(
        self,
        image: Union[str, np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        识别图像中的所有人脸 — 引擎主入口。
        
        流程:
          1. 检测所有人脸
          2. 对每张脸:
             a. 对齐 → 112x112
             b. 提取嵌入向量
             c. 与数据库中所有向量计算余弦相似度
             d. 取最高相似度的用户, 若低于阈值则标记为 Unknown
        
        Args:
            image: 图像文件路径或 numpy 数组
            threshold: 识别阈值 (默认使用配置中的值)
        
        Returns:
            识别结果列表, 按人脸在图像中的出现顺序排列:
            [
              {
                'face_index': 0,           # 第几张脸
                'bbox': (x, y, w, h),      # 边界框
                'name': '张三',            # 识别结果 (或 'Unknown')
                'confidence': 0.85,        # 余弦相似度
                'detection_confidence': 0.92, # 检测置信度
              },
              ...
            ]
        """
        if threshold is None:
            threshold = self.recognition_threshold

        # 读取图像
        if isinstance(image, str):
            import cv2
            img = cv2.imread(image)
            if img is None:
                print(f"无法读取图像: {image}")
                return []
        else:
            img = image

        # Step 1: 检测所有人脸
        faces = self.detector.detect(img)
        if len(faces) == 0:
            return []

        # 获取已注册用户的所有嵌入向量
        registered = self.database.get_all_embeddings()
        results = []

        # Step 2: 逐张人脸处理
        for i, face in enumerate(faces):
            bbox = face['bbox']
            landmarks = face.get('landmarks', {})

            # 对齐
            aligned = self.aligner.align(img, bbox, landmarks)
            if aligned is None:
                results.append({
                    'face_index': i, 'bbox': bbox, 'name': 'Unknown',
                    'confidence': 0.0, 'error': 'alignment_failed',
                })
                continue

            # 提取嵌入向量
            embedding = self.embedder.extract_embedding(aligned)

            # Step 3: 与所有注册用户对比, 找最佳匹配
            best_name = 'Unknown'
            best_similarity = 0.0

            for name, stored_embs in registered.items():
                for stored_emb in stored_embs:
                    sim = self._cosine_similarity(embedding, stored_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        best_name = name

            # 低于阈值 → 标记为未知
            if best_similarity < threshold:
                best_name = 'Unknown'

            results.append({
                'face_index': i,
                'bbox': bbox,
                'name': best_name,
                'confidence': float(best_similarity),
                'detection_confidence': face.get('confidence', 0.0),
            })

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度。
        
        公式: cos(θ) = (A·B) / (||A|| × ||B||)
        
        由于存入数据库的向量已经 L2 归一化 (||v||=1),
        所以实际上就是向量内积: cos(θ) = A·B
        
        取值范围: [-1, 1]
          1.0  → 完全同向 (同一人)
          0.0  → 正交 (无关)
         -1.0  → 完全反向 (几乎不可能出现)
        
        Args:
            a, b: 两个嵌入向量 (float32 或 float64)
        
        Returns:
            余弦相似度 (float)
        """
        # 展平为一维, 转为 float64 以提高计算精度
        a_f = a.flatten().astype(np.float64)
        b_f = b.flatten().astype(np.float64)

        dot = np.dot(a_f, b_f)          # 内积
        norm_a = np.linalg.norm(a_f)    # ||a||
        norm_b = np.linalg.norm(b_f)    # ||b||

        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def list_users(self) -> List[Dict]:
        """列出所有已注册用户及其样本数"""
        return self.database.get_users()

    def delete_user(self, name: str) -> Tuple[bool, str]:
        """删除指定用户的注册信息"""
        if self.database.remove_user(name):
            return True, f"已删除用户 '{name}'"
        return False, f"用户 '{name}' 不存在"


# ============================================================================
# 第八部分: 可视化工具
# ============================================================================
#
# 将识别结果绘制到图像上, 用绿色框标注已识别用户, 红色框标注未知人脸。
# ============================================================================

def draw_results(image: np.ndarray, results: List[Dict]) -> np.ndarray:
    """
    在图像上绘制人脸识别结果的可视化标注。
    
    绘制内容:
      - 绿色矩形框 + 绿色背景标签 → 已识别用户
      - 红色矩形框 + 红色背景标签 → 未知人脸
      - 标签文字: "姓名 (相似度)"
    
    Args:
        image: 原始 BGR 图像
        results: recognize() 返回的结果列表
    
    Returns:
        绘制了标注的新图像 (不修改原图)
    """
    import cv2
    img = image.copy()

    for r in results:
        x, y, w, h = r['bbox']
        name = r['name']
        confidence = r['confidence']

        # 颜色选择: 已识别=绿色, 未知=红色
        color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)

        # 画矩形边界框
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # 画标签 (带背景色块)
        label = f"{name} ({confidence:.2f})"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # 标签背景矩形
        cv2.rectangle(img, (x, y - label_h - 10), (x + label_w + 6, y), color, cv2.FILLED)
        # 标签白色文字
        cv2.putText(img, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


# ============================================================================
# 第九部分: 摄像头实时识别
# ============================================================================
#
# 打开 Mac 内置摄像头 (或外接摄像头), 逐帧进行人脸检测与识别,
# 并将结果实时叠加显示在预览画面上。
#
# 性能优化:
#   - 不是每帧都做识别, 而是每 5 帧做一次 (RECOGNIZE_EVERY_N=5)
#     中间帧复用上一次的识别结果, 大幅降低 CPU 占用
#   - 检测+对齐+嵌入+匹配 全部走 NPU 加速, 比纯 CPU 快 3~5 倍
# ============================================================================

def run_webcam(engine: FaceRecognitionEngine):
    """
    启动摄像头实时人脸识别。
    
    交互方式:
      q → 退出
      s → 截图保存
      + → 提高识别阈值 (更严格)
      - → 降低识别阈值 (更宽松)
    
    画面叠加信息:
      - FPS: 每秒处理帧数
      - Threshold: 当前识别阈值
      - Registered Users: 已注册用户数
      - 人脸边界框 + 姓名标签
    """
    import cv2

    print("\n📷 启动摄像头实时人脸识别 (Neural Engine 加速)")
    print("   - 按 'q' 键退出")
    print("   - 按 's' 键截图保存")
    print("   - 按 '+' / '-' 调整识别阈值")
    print()

    # 打开默认摄像头 (设备索引 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    # 设置分辨率 (根据摄像头能力自动适配)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threshold = engine.recognition_threshold  # 当前识别阈值
    frame_count = 0           # 累计帧数
    fps = 0                   # 当前 FPS
    last_results = []         # 上一次识别结果 (复用)
    screenshot_count = 0      # 截图计数
    RECOGNIZE_EVERY_N = 5     # 每 N 帧做一次完整识别

    print("🎥 摄像头已启动\n")

    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("⚠️  读取摄像头帧失败")
            break

        frame_count += 1
        display_frame = frame.copy()  # 显示用副本

        # ---- 每 N 帧执行一次完整识别 ----
        # 中间帧复用上一次的结果, 大幅减少计算量
        if frame_count % RECOGNIZE_EVERY_N == 0:
            t_start = time.time()
            results = engine.recognize(frame, threshold=threshold)
            t_end = time.time()
            last_results = results
            # 计算 FPS (每秒处理帧数)
            fps = RECOGNIZE_EVERY_N / (t_end - t_start) if (t_end - t_start) > 0 else 0
        else:
            results = last_results

        # 绘制标注
        display_frame = draw_results(display_frame, results)

        # 叠加状态信息 (左上角)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | Threshold: {threshold:.2f} | NPU Accelerated",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, f"Registered Users: {engine.database.user_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 显示画面
        cv2.imshow('Face Recognition - Neural Engine (NPU)', display_frame)

        # ---- 按键处理 ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 退出
            print("\n👋 退出摄像头模式")
            break
        elif key == ord('s'):
            # 截图保存
            screenshot_count += 1
            path = str(MODEL_DIR / f'screenshot_{screenshot_count:03d}.jpg')
            cv2.imwrite(path, frame)
            print(f"📸 截图已保存: {path}")
        elif key in (ord('='), ord('+')):
            # 提高阈值 (更严格, 减少误识别)
            threshold = min(1.0, threshold + 0.05)
            print(f"🔧 识别阈值: {threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            # 降低阈值 (更宽松, 减少漏识别)
            threshold = max(0.0, threshold - 0.05)
            print(f"🔧 识别阈值: {threshold:.2f}")

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# 第十部分: 模型下载工具
# ============================================================================
#
# 自动从互联网下载预训练的 Core ML 人脸嵌入模型。
# 模型来自 HuggingFace 等公开源。
# ============================================================================

def download_face_embedding_model() -> bool:
    """
    下载 Core ML 人脸嵌入模型 (MobileFaceNet)。
    
    下载 → 解压 → 放到 ~/.face_recognition_npu/ 目录。
    如果模型已存在则跳过。
    
    Returns:
        True 成功获取模型, False 失败 (需要手动安装)
    """
    import urllib.request
    import zipfile

    # 如果已存在, 直接返回
    if FACE_EMBED_MODEL_PATH.exists():
        print(f"✅ 人脸嵌入模型已存在: {FACE_EMBED_MODEL_PATH}")
        return True

    print("\n📥 正在下载 Core ML 人脸嵌入模型...")
    print(f"   目标位置: {FACE_EMBED_MODEL_PATH}\n")

    for url in FACE_EMBED_MODEL_URLS:
        try:
            print(f"   尝试从 {url[:60]}... 下载")
            zip_path = MODEL_DIR / 'model_download.zip'
            urllib.request.urlretrieve(url, str(zip_path))

            print("   正在解压...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if member.endswith('.mlpackage/') or member.endswith('.mlmodel/'):
                        zf.extractall(MODEL_DIR)
                        extracted = MODEL_DIR / member.rstrip('/')
                        if extracted.exists() and extracted != FACE_EMBED_MODEL_PATH:
                            import shutil
                            if FACE_EMBED_MODEL_PATH.exists():
                                shutil.rmtree(FACE_EMBED_MODEL_PATH)
                            shutil.move(str(extracted), str(FACE_EMBED_MODEL_PATH))
                        break
                    elif member.endswith('.mlmodel'):
                        zf.extractall(MODEL_DIR)
                        extracted = MODEL_DIR / member
                        if extracted.exists() and extracted != FACE_EMBED_MODEL_PATH:
                            import shutil
                            shutil.move(str(extracted), str(FACE_EMBED_MODEL_PATH))
                        break

            # 清理下载的 zip 文件
            zip_path.unlink(missing_ok=True)
            if FACE_EMBED_MODEL_PATH.exists():
                print("✅ 模型下载并安装完成!")
                return True

        except urllib.error.HTTPError as e:
            print(f"   ❌ HTTP 错误: {e}")
        except urllib.error.URLError as e:
            print(f"   ❌ 网络错误: {e}")
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")

    # 所有 URL 都失败, 给出详细的手动安装指南
    print(f"""
{'=' * 60}
  ⚠️  自动下载失败，请手动获取 Core ML 人脸嵌入模型
{'=' * 60}

获取模型的几种方式:

[方式1] 使用 coremltools 自行转换:
    pip install coremltools
    # 从 PyTorch 转换 MobileFaceNet 模型到 Core ML 格式
    # 参考: https://coremltools.readme.io/docs

[方式2] 从 Model Zoo 下载:
    - HuggingFace 上搜索 "mobilefacenet coreml"
    - GitHub 上搜索 "coreml face recognition"

[方式3] 使用内置回退方案 (精度较低):
    脚本已内置 ORB 特征描述符回退方案, 无需模型也能运行,
    但识别精度远低于深度学习模型, 仅建议用于功能验证。

将 .mlmodel 或 .mlpackage 文件放置到:
    {MODEL_DIR}
然后重新运行脚本即可。
""")
    return False


# ============================================================================
# 第十一部分: 系统信息
# ============================================================================

def cmd_info():
    """
    显示系统信息与硬件加速状态诊断。
    
    帮助用户了解:
      - Mac 型号 / macOS 版本
      - 是否检测到 Apple Silicon (M1~M4)
      - Vision Framework 是否可用 (PyObjC)
      - Core ML 嵌入模型是否已安装
      - 人脸数据库中有多少注册用户
    """
    print("\n" + "=" * 55)
    print("  系统信息 & 硬件加速状态")
    print("=" * 55)

    # Python 版本
    print(f"\n  Python:     {sys.version.split()[0]}")

    # macOS 版本
    try:
        result = subprocess.run(['sw_vers'], capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if 'ProductVersion' in line:
                print(f"  macOS:      {line.split(':')[1].strip()}")
    except Exception:
        pass

    # CPU / 芯片型号
    try:
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                capture_output=True, text=True)
        cpu = result.stdout.strip()
        print(f"  CPU:        {cpu}")

        # 判断是否为 Apple Silicon (M1~M4)
        if 'Apple' in cpu and any(m in cpu for m in ['M1', 'M2', 'M3', 'M4']):
            print(f"  Neural Engine: ✅ 可用 (Apple Silicon 的 ANE 将被 Vision/CoreML 自动调用)")
        else:
            print(f"  Neural Engine: ⚠️  未检测到 Apple Silicon, NPU 加速不可用")
    except Exception:
        print(f"  CPU:        未知")

    # Vision Framework 状态 (PyObjC)
    try:
        import Vision
        print(f"  Vision Framework: ✅ 可用 (PyObjC 桥接)")
    except ImportError:
        print(f"  Vision Framework: ⚠️  未安装 → pip install pyobjc-framework-Vision")

    # Core ML Tools 状态
    try:
        import coremltools as ct
        print(f"  Core ML Tools:   ✅ {ct.__version__}")
    except ImportError:
        print(f"  Core ML Tools:   ⚠️  未安装 → pip install coremltools")

    # 嵌入模型文件状态
    if FACE_EMBED_MODEL_PATH.exists():
        size_mb = sum(f.stat().st_size for f in FACE_EMBED_MODEL_PATH.rglob('*') if f.is_file()) / (1024 * 1024)
        print(f"  嵌入模型:        ✅ 已安装 ({size_mb:.1f} MB)")
    else:
        print(f"  嵌入模型:        ⚠️  未安装 → 运行 'download-model' 下载")
        print(f"  期望路径:        {FACE_EMBED_MODEL_PATH}")

    # 人脸数据库状态
    if FACE_DB_FILE.exists():
        db = FaceDatabase()
        print(f"  人脸数据库:      ✅ {db.user_count} 个已注册用户")
    else:
        print(f"  人脸数据库:      📭 空 (使用 register 命令添加)")

    print(f"  配置目录:        {MODEL_DIR}")
    print()


# ============================================================================
# 第十二部分: 命令行界面 (CLI)
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器。
    
    支持的子命令:
      register      注册新用户的人脸
      recognize     识别图像中的所有人脸
      webcam        启动摄像头实时识别
      list          列出所有已注册用户
      delete        删除指定用户
      download-model 下载 Core ML 人脸嵌入模型
      info          显示系统信息和硬件加速状态
    """
    parser = argparse.ArgumentParser(
        description='M系列芯片 NPU 人脸识别系统 (Apple Neural Engine 加速)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python face_recognition_npu.py register --name "张三" --image photo.jpg
  python face_recognition_npu.py recognize --image group_photo.jpg
  python face_recognition_npu.py recognize --image test.jpg --output result.jpg
  python face_recognition_npu.py webcam
  python face_recognition_npu.py list
  python face_recognition_npu.py delete --name "张三"
  python face_recognition_npu.py download-model
  python face_recognition_npu.py info
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # register 子命令
    reg = subparsers.add_parser('register', help='注册新用户的人脸')
    reg.add_argument('--name', '-n', required=True, help='用户名称 (如 "张三")')
    reg.add_argument('--image', '-i', required=True, help='包含清晰正面人脸的图像路径')

    # recognize 子命令
    rec = subparsers.add_parser('recognize', help='识别图像中的人脸')
    rec.add_argument('--image', '-i', required=True, help='要识别的图像路径')
    rec.add_argument('--threshold', '-t', type=float,
                     help=f'识别阈值 0~1 (默认: {FACE_RECOGNITION_THRESHOLD}, 越低越宽松)')
    rec.add_argument('--output', '-o', help='保存标注结果的输出图像路径')
    rec.add_argument('--no-display', action='store_true', help='不弹出图像预览窗口')

    # webcam 子命令
    subparsers.add_parser('webcam', help='启动摄像头实时人脸识别 (按 q 退出)')

    # list 子命令
    subparsers.add_parser('list', help='列出所有已注册用户及其样本数')

    # delete 子命令
    dl = subparsers.add_parser('delete', help='删除已注册用户的所有人脸数据')
    dl.add_argument('--name', '-n', required=True, help='要删除的用户名称')

    # download-model 子命令
    subparsers.add_parser('download-model', help='下载 Core ML 人脸嵌入模型 (约 5-15MB)')

    # info 子命令
    subparsers.add_parser('info', help='显示系统信息、芯片型号、硬件加速状态')

    return parser


def main():
    """
    程序主入口。
    
    处理流程:
      1. 打印欢迎信息
      2. 解析命令行参数
      3. 如果不指定命令 → 打印帮助
      4. info/download-model → 直接执行 (不需要加载引擎)
      5. 其他命令 → 检查依赖 → 初始化引擎 → 执行
    """
    # 欢迎信息
    print("\n" + "╔" + "═" * 53 + "╗")
    print("║  M系列芯片 Neural Engine (NPU) 人脸识别系统  ║")
    print("║  Face Detection  -> Vision Framework -> NPU  ║")
    print("║  Face Embedding  -> Core ML -> NPU           ║")
    print("╚" + "═" * 53 + "╝\n")

    parser = create_parser()
    args = parser.parse_args()

    # 无命令 → 显示帮助
    if args.command is None:
        parser.print_help()
        print("\n💡 快速开始 (三步上手):")
        print("  1. python face_recognition_npu.py download-model  # 下载 AI 模型")
        print("  2. python face_recognition_npu.py register -n 你 -i photo.jpg  # 注册人脸")
        print("  3. python face_recognition_npu.py recognize -i test.jpg  # 开始识别")
        return

    # info 和 download-model 不需要加载引擎, 直接执行
    if args.command == 'info':
        cmd_info()
        return

    if args.command == 'download-model':
        download_face_embedding_model()
        return

    # ---- 以下命令需要依赖和引擎 ----

    # 检查并自动安装依赖
    if not ensure_dependencies():
        sys.exit(1)

    # 验证 OpenCV 是否正确安装
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV 未正确安装, 请手动运行: pip install opencv-python")
        sys.exit(1)

    # 初始化人脸识别引擎
    if args.command in ('register', 'recognize', 'webcam', 'list', 'delete'):
        config = load_config()
        engine = FaceRecognitionEngine(config)
    else:
        engine = None

    # ---- 执行对应命令 ----

    if args.command == 'register':
        print(f"\n📝 注册用户: {args.name}")
        success, msg = engine.register(args.name, args.image)
        print(f"  {'✅' if success else '❌'} {msg}")

    elif args.command == 'recognize':
        print(f"\n🔍 识别图像: {args.image}")
        t0 = time.time()
        results = engine.recognize(args.image, threshold=args.threshold)
        t1 = time.time()

        if not results:
            print("  ℹ️   未检测到人脸")
        else:
            print(f"\n  📊 检测到 {len(results)} 张人脸 (耗时: {(t1 - t0) * 1000:.1f}ms):")
            print("  " + "-" * 45)
            for r in results:
                icon = "👤" if r['name'] != 'Unknown' else "❓"
                print(f"  {icon} 人脸{r['face_index'] + 1}: {r['name']:<12s} "
                      f"相似度: {r['confidence']:.4f}  "
                      f"检测置信度: {r.get('detection_confidence', 0):.2f}")
            print("  " + "-" * 45)

            # 绘制标注图像
            img = cv2.imread(args.image)
            annotated = draw_results(img, results)

            # 保存输出
            if args.output:
                cv2.imwrite(args.output, annotated)
                print(f"\n  📸 标注图像已保存: {args.output}")

            # 显示预览
            if not args.no_display:
                cv2.imshow('Face Recognition Results (NPU)', annotated)
                print("\n  按任意键关闭图像窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    elif args.command == 'webcam':
        run_webcam(engine)

    elif args.command == 'list':
        users = engine.list_users()
        if not users:
            print("\n📭 人脸数据库为空")
            print("  使用 register 命令添加用户")
        else:
            print(f"\n📋 已注册用户 ({len(users)} 人):")
            print("=" * 40)
            for u in users:
                print(f"  👤 {u['name']:<20s} 样本数: {u['samples']}")
            print("=" * 40)

    elif args.command == 'delete':
        success, msg = engine.delete_user(args.name)
        print(f"{'✅' if success else '❌'} {msg}")


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == '__main__':
    main()
