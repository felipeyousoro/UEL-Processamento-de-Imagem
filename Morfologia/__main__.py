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


def get_von_neumann_neighborhood(img, x, y) -> list[int, int]:
    max_x, max_y = img.shape

    neighborhood = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0) or (i != 0 and j != 0) or (x + i < 0 or x + i >= max_x) or (
                    y + j < 0 or y + j >= max_y):
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


def extract_frontier():
    erosion_img = erosion()

    frontier = PIROCA.copy()

    return frontier - erosion_img


def hole_filler():
    frontier = extract_frontier()

    hole_fill = frontier.copy()
    for i in range(frontier.shape[0]):
        for j in range(frontier.shape[1]):
            if frontier[i, j] == 0:
                hole_fill[i, j] = 1
            else:
                hole_fill[i, j] = 0

    return hole_fill


def fill_hole_dfs(x, y, hole, filled_hole, visited):
    if visited[x, y] == 1:
        return

    visited[x, y] = 1
    filled_hole[x, y] = 1

    neighborhood = get_von_neumann_neighborhood(hole, x, y)

    for i, j in neighborhood:
        if hole[i, j] == 1:
            fill_hole_dfs(i, j, hole, filled_hole, visited)


def fill_holes():
    frontier = extract_frontier()
    hole_fill = hole_filler()

    filled_hole = np.zeros(hole_fill.shape, dtype=np.uint8)

    start_x = 4
    start_y = 0

    visited = np.zeros(hole_fill.shape, dtype=np.uint8)

    fill_hole_dfs(start_x, start_y, hole_fill, filled_hole, visited)

    return filled_hole + frontier


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

    frontier_img = extract_frontier()
    frontier_img[frontier_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/frontier.png', frontier_img)

    hole_fill_img = hole_filler()
    hole_fill_img[hole_fill_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/hole.png', hole_fill_img)

    filled_hole_img = fill_holes()
    filled_hole_img[filled_hole_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/filled_hole.png', filled_hole_img)
