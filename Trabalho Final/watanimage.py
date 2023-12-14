import numpy as np
import cv2 as cv


def binarize_image(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    copy = (copy > 127).astype(np.uint8)
    return copy


def erosion(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.erode(copy, np.ones((3, 3), dtype=np.uint8))


def dilation(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.dilate(copy, np.ones((3, 3), dtype=np.uint8))


def opening(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.morphologyEx(copy, cv.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))


def closing(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.morphologyEx(copy, cv.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.equalizeHist(copy)


def sobel_filter(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    copy = cv.Sobel(copy, cv.CV_64F, 1, 1, ksize=3)
    copy = np.absolute(copy)
    copy = np.uint8(copy)
    return copy


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.GaussianBlur(copy, (5, 5), 0)
