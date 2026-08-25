import logging

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    """OCR 引擎不可用（模型未下载或初始化失败）。"""


def _parse_ocr_result(result) -> str:
    """解析 PaddleOCR 2.x 返回的嵌套结果 [[[box, (text, conf)], ...], ...]。"""
    lines = []
    for page in result or []:
        for line in page or []:
            if len(line) >= 2 and line[1]:  # [box, (text, conf)]
                lines.append(str(line[1][0]))
    return "\n".join(lines).strip()


class OCRClient:
    """PaddleOCR 封装。构造时懒加载模型；失败降级为不可用，不阻塞流水线。"""

    def __init__(self, engine=None, lang: str = "ch"):
        self._engine = engine  # 测试注入；None 时懒加载 PaddleOCR
        self._lang = lang
        self._injected = engine is not None
        self._loaded = engine is not None

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from paddleocr import PaddleOCR
            self._engine = PaddleOCR(
                use_angle_cls=True, lang=self._lang, show_log=False)
            self._loaded = True
        except Exception as exc:  # 模型下载失败 / 无 GPU / 依赖缺失
            logger.warning("PaddleOCR 初始化失败: %s", exc)
            self._loaded = False

    def extract_text(self, image_bytes: bytes) -> str:
        """对图片字节做 OCR，返回拼接文本；无文字返回空串。"""
        self._ensure_loaded()
        if self._engine is None:
            raise OCRUnavailableError("OCR 引擎不可用")
        if self._injected:
            # 测试注入的引擎：直接接收原始字节，返回字符串
            return str(self._engine.ocr(image_bytes)).strip()
        import numpy as np
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = self._engine.ocr(np.array(img), cls=True)
        return _parse_ocr_result(result)
