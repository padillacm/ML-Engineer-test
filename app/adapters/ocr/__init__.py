"""Tesseract OCR adapter: renders PDF pages, flattens illumination and reads words with boxes."""

from app.adapters.ocr.engine import TesseractOcrConfig, TesseractOcrEngine

__all__ = ["TesseractOcrConfig", "TesseractOcrEngine"]
