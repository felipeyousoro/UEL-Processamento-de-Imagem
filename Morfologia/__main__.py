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
    erosion_img = cv.erode(PIROCA, KERNEL, iterations=1)

    return erosion_img


def opening():
    erosion_img = erosion()
    opening_img = np.zeros(erosion_img.shape, dtype=np.uint8)

    for i in range(erosion_img.shape[0]):
        for j in range(erosion_img.shape[1]):
            if erosion_img[i, j] == 1:
                neighborhood = get_moore_neighborhood(erosion_img, i, j)

                for x, y in neighborhood:
                    opening_img[x, y] = 1

    return opening_img

def dilation():
    dilation_img = cv.dilate(PIROCA, KERNEL, iterations=1)

    return dilation_img

def closing():
    dilation_img = dilation()
    closing = np.zeros(dilation_img.shape, dtype=np.uint8)

    for i in range(dilation_img.shape[0]):
        for j in range(dilation_img.shape[1]):
            if dilation_img[i, j] == 1:
                neighborhood = get_moore_neighborhood(dilation_img, i, j)

                if all(dilation_img[x, y] == 1 for x, y in neighborhood):
                    closing[i, j] = 1

    return closing


if __name__ == '__main__':
    original_img = PIROCA.copy()
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