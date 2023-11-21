import math as m
import cv2 as cv
import numpy as np

# PATHING DAS IMAGES
OUTPUT_FOLDER = 'outputs'

PIROCA = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

KERNEL = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))


def get_moore_neighborhood(img, x, y) -> list[int, int]:
    max_x, max_y = img.shape

    neighborhood = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0) or (x + i < 0 or x + i >= max_x) or (y + j < 0 or y + j >= max_y):
                continue

            neighborhood.append((x + i, y + j))

    return neighborhood

def erosion():
    img = PIROCA
    kernel = KERNEL

    erosion = cv.erode(img, kernel, iterations=1)

    return erosion


def opening():
    img = PIROCA
    kernel = KERNEL

    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    return opening

def dilation():
    img = PIROCA
    kernel = KERNEL

    dilation = cv.dilate(img, kernel, iterations=1)

    return dilation

def closing():
    img = PIROCA
    kernel = KERNEL

    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    return closing


if __name__ == '__main__':
    original_img = PIROCA
    original_img[original_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/original.png', original_img)

    erosion_img = erosion()
    erosion_img[erosion_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/erosion.png', erosion_img)

    opening_img = opening()
    opening_img[opening_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/opening.png', opening_img)

    dilation_img = dilation()
    dilation_img[dilation_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/dilation.png', dilation_img)

    closing_img = closing()
    closing_img[closing_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/closing.png', closing_img)