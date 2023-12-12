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


def bilateral_filter(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.bilateralFilter(copy, 9, 75, 75)

def sobel_filter(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    copy = cv.Sobel(copy, cv.CV_64F, 1, 1, ksize=3)
    copy = np.absolute(copy)
    copy = np.uint8(copy)
    return copy


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return cv.GaussianBlur(copy, (5, 5), 0)


def fourier_transform(img: np.ndarray) -> np.ndarray:
    copy = img.copy()
    return np.fft.fft2(copy)


def get_connected_components(img: np.ndarray) -> list:
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)

    components_coordinates = []

    for label in range(1, num_labels):  # Start from 1 to exclude the background (label 0)
        component_coordinates = np.column_stack(np.where(labels == label))
        components_coordinates.append(component_coordinates)

    return components_coordinates
