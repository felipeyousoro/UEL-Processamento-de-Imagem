import math as m
import cv2 as cv
import numpy as np

import watanimage

# PATHING DAS IMAGENS
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'


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
    img = cv.imread(INPUT_IMAGE, cv.IMREAD_GRAYSCALE)
    img = watanimage.binarize_image(img)

    erosion_img = cv.erode(img, watanimage.MOORE_KERNEL, iterations=1)
    erosion_img[erosion_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/erosion.png', erosion_img)

    opening_img = watanimage.opening(img, watanimage.MOORE_KERNEL)
    opening_img[opening_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/opening.png', opening_img)

    closing_img = watanimage.closing(img, watanimage.MOORE_KERNEL)
    closing_img[closing_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/closing.png', closing_img)

    frontier_img = watanimage.extract_frontier(img)
    frontier_img[frontier_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/frontier.png', frontier_img)

    fillings = watanimage.extract_frontier(img)
    fillings = watanimage.invert_image(fillings)
    fillings[fillings == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/fillings.png', fillings)

    filled_img = watanimage.fill_holes(frontier_img, fillings, 64, 164)
    filled_img[filled_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/filled_image.png', filled_img)

    c_comp_img = watanimage.get_connected_components(img, watanimage.MOORE_KERNEL, 64, 164)
    c_comp_img[c_comp_img == 1] = 255
    cv.imwrite(f'{OUTPUT_FOLDER}/connected_components.png', c_comp_img)

