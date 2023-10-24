import math as m
import cv2 as cv
import numpy as np

# PATHING DAS IMAGES
INPUT_IMAGE = 'rozemyne.png'
OUTPUT_FOLDER = 'outputs'

MEAN_PIXELS = 3
WEIGHTS = np.array([
    [1, 2, 1],
    [2, 4, 2],
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
                sum += np.uint32(image[i][j]) * WEIGHTS[i - x + shift][j - y + shift]
                weight_count += WEIGHTS[i - x + shift][j - y + shift]

    return np.array(sum // weight_count, dtype=np.uint8)


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


if __name__ == '__main__':
    # image has color
    input_image = cv.imread(INPUT_IMAGE, cv.IMREAD_COLOR)

    mean_image = mean_filter(input_image)
    cv.imwrite(f'{OUTPUT_FOLDER}/mean_image.png', mean_image)

    weighted_mean_image = weighted_mean_filter(input_image)
    cv.imwrite(f'{OUTPUT_FOLDER}/weighted_mean_image.png', weighted_mean_image)
