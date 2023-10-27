import math as m
import cv2 as cv
import numpy as np

# PATHING DAS IMAGES
INPUT_IMAGE = 'rozenoise.png'
OUTPUT_FOLDER = 'outputs'

MEAN_PIXELS = 3
WEIGHTS = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

SOBEL_VERTICAL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

SOBEL_HORIZONTAL = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


def mean_pixel(image: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width, _ = image.shape
    sum = np.zeros(3, dtype=np.uint32)

    shift: int = MEAN_PIXELS // 2

    pixels_count = 0
    for i in range(x - shift, x + shift + 1):
        for j in range(y - shift, y + shift + 1):
            if 0 <= i < height and 0 <= j < width:
                sum += image[i][j]
                pixels_count += 1

    return np.array(sum // pixels_count, dtype=np.uint8)


def weighted_mean_pixel(image: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width, _ = image.shape
    sum = np.zeros(3, dtype=np.uint64)

    shift: int = MEAN_PIXELS // 2

    weight_count = 0
    for i in range(x - shift, x + shift + 1):
        for j in range(y - shift, y + shift + 1):
            if 0 <= i < height and 0 <= j < width:
                sum += np.uint64(image[i][j]) * WEIGHTS[i - x + shift][j - y + shift]
                weight_count += WEIGHTS[i - x + shift][j - y + shift]

    return np.array(sum // weight_count, dtype=np.uint8)


def sobel_pixel(image: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width, _ = image.shape
    sum_hor = np.zeros(3, dtype=np.int64)
    sum_ver = np.zeros(3, dtype=np.int64)

    shift: MEAN_PIXELS = 3 // 2

    weight_count_vert = 0
    weight_count_hor = 0
    for i in range(x - shift, x + shift + 1):
        for j in range(y - shift, y + shift + 1):
            if 0 <= i < height and 0 <= j < width:
                sum_hor += np.int64(image[i][j]) * SOBEL_HORIZONTAL[i - x + shift][j - y + shift]
                sum_ver += np.int64(image[i][j]) * SOBEL_VERTICAL[i - x + shift][j - y + shift]
                weight_count_vert += SOBEL_VERTICAL[i - x + shift][j - y + shift]
                weight_count_hor += SOBEL_HORIZONTAL[i - x + shift][j - y + shift]

    sum_hor = sum_hor // weight_count_hor
    sum_ver = sum_ver // weight_count_vert

    return np.array(sum_hor + sum_ver, dtype=np.uint8)


def mean_filter(image: np.ndarray) -> np.ndarray:
    new_image: np.ndarray = np.zeros(image.shape)
    height, width, _ = image.shape

    for x in range(height):
        for y in range(width):
            new_image[x][y] = mean_pixel(image, x, y)

    return new_image


def weighted_mean_filter(image: np.ndarray) -> np.ndarray:
    new_image: np.ndarray = np.zeros(image.shape)
    height, width, _ = image.shape

    for x in range(height):
        for y in range(width):
            new_image[x][y] = weighted_mean_pixel(image, x, y)

    return new_image


def sobel_filter(image: np.ndarray) -> np.ndarray:

    edge_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    edge_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    edge = np.sqrt(np.square(edge_x) + np.square(edge_y))

    return edge

if __name__ == '__main__':
    # image has color
    input_image = cv.imread(INPUT_IMAGE, cv.IMREAD_COLOR)

    mean_image = mean_filter(input_image)
    cv.imwrite(f'{OUTPUT_FOLDER}/mean_image.png', mean_image)

    weighted_mean_image = weighted_mean_filter(input_image)
    cv.imwrite(f'{OUTPUT_FOLDER}/weighted_mean_image.png', weighted_mean_image)

    edge = sobel_filter(input_image)
    edge = np.array(edge, dtype=np.uint8)

    new_image = cv.add(edge, input_image)

    cv.imwrite(f'{OUTPUT_FOLDER}/sobel_filter.png', edge)

    enhanced_image = cv.addWeighted(input_image, 0.7, edge, 0.3, 0)

    cv.imwrite(f'{OUTPUT_FOLDER}/sobel.png', enhanced_image)

