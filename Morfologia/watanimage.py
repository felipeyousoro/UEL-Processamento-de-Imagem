import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

MOORE_KERNEL = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]], dtype=np.uint8)

VON_NEUMANN_KERNEL = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]], dtype=np.uint8)


def get_moore_neighborhood(img, x, y) -> list[int, int]:
    max_x, max_y = img.shape

    neighborhood: list[int, int] = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0) or (x + i < 0 or x + i >= max_x) or (y + j < 0 or y + j >= max_y):
                continue

            neighborhood.append([int(x + i), int(y + j)])

    return neighborhood


def get_kernel_neighborhood(img, x, y, kernel: np.ndarray) -> list[int, int]:
    max_x, max_y = img.shape

    neighborhood: list[int, int] = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if kernel[i][j] == 0:
                continue
            if (x + i < 0 or x + i >= max_x) or (y + j < 0 or y + j >= max_y):
                continue

            neighborhood.append([int(x + i), int(y + j)])

    return neighborhood


def are_images_equal(img1: np.ndarray, img2: np.ndarray) -> bool:
    x1, y1 = img1.shape
    x2, y2 = img2.shape

    if x1 != x2 or y1 != y2:
        return False

    for i in range(0, x1):
        for j in range(0, y1):
            if img1[i][j] != img2[i][j]:
                return False

    return True


def opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    erosion = cv.erode(img, kernel, iterations=1)
    opening_img = np.zeros(erosion.shape, dtype=np.uint8)

    for i in range(erosion.shape[0]):
        for j in range(erosion.shape[1]):
            if erosion[i, j] == 1:
                neighborhood = get_kernel_neighborhood(erosion, i, j, kernel)

                for x, y in neighborhood:
                    opening_img[x, y] = 1

    return opening_img


def closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    dilation = cv.dilate(img, kernel, iterations=1)
    closing_img = np.zeros(dilation.shape, dtype=np.uint8)

    for i in range(dilation.shape[0]):
        for j in range(dilation.shape[1]):
            if dilation[i, j] == 1:
                neighborhood = get_kernel_neighborhood(dilation, i, j, kernel)

                if all(dilation[x, y] == 1 for x, y in neighborhood):
                    closing_img[i, j] = 1

    return closing_img


def extract_frontier(img: np.ndarray) -> np.ndarray:
    erosion = cv.erode(img, MOORE_KERNEL, iterations=1)

    frontier_img = img.copy() - erosion

    return frontier_img


def fill_holes(frontier: np.ndarray, filling: np.ndarray, x: int, y: int) -> np.ndarray:
    filled_img = np.zeros(frontier.shape, dtype=np.uint8)
    filled_img[x, y] = 1

    while True:
        k_iteration = filled_img.copy()
        k_iteration = cv.dilate(k_iteration, VON_NEUMANN_KERNEL, iterations=1)

        k_iteration = k_iteration & filling

        if not are_images_equal(k_iteration, filled_img):
            filled_img = k_iteration
        else:
            break

    return filled_img + frontier


def get_connected_components(img: np.ndarray, kernel: np.ndarray, x: int, y: int) -> np.ndarray:
    c_components_image = np.zeros(img.shape, dtype=np.uint8)
    c_components_image[x, y] = 1

    while True:
        k_iteration = c_components_image.copy()
        k_iteration = cv.dilate(k_iteration, kernel, iterations=1)

        k_iteration = k_iteration & img

        if not are_images_equal(k_iteration, c_components_image):
            c_components_image = k_iteration
        else:
            break

    return c_components_image


def binarize_image(img: np.ndarray) -> np.ndarray:
    bin_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] > 127:
                bin_img[i][j] = 1
            else:
                bin_img[i][j] = 0

    return bin_img


def invert_image(img: np.ndarray) -> np.ndarray:
    inverted_img = np.zeros(img.shape, dtype=np.uint8)

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] == 0:
                inverted_img[i][j] = 1
            else:
                inverted_img[i][j] = 0

    return inverted_img
