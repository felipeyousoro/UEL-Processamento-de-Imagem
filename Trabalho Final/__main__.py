import math as m
import cv2 as cv
import numpy as np
import watanimage

# PATHING DAS IMAGES
INPUT_FOLDER = 'inputs'
OUTPUT_FOLDER = 'outputs'


def save_images(imgs, name):
    for i in range(len(imgs)):
        cv.imwrite(OUTPUT_FOLDER + '/' + name + str(i + 1) + '.png', imgs[i])


def mark_components(colored_image: np.ndarray, components: list) -> np.ndarray:
    marked_image = colored_image.copy()

    for component_coordinates in components:
        for pixel_coordinates in component_coordinates:
            y, x = pixel_coordinates
            marked_image[y, x] = (0, 255, 0)  # You can adjust the color

    cv.imshow('Marked Components', marked_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return marked_image


def mark_rectangular_contours(colored_image: np.ndarray, binarized_image: np.ndarray) -> np.ndarray:
    marked_image = colored_image.copy()

    contours, hierarchy = cv.findContours(binarized_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)

        # Check if the polygon has four vertices (a rectangle)
        if len(approx) != 4:
            cv.drawContours(marked_image, [contour], -1, (127, 0, 255), 3)
        elif len(approx) == 4:
            cv.drawContours(marked_image, [contour], -1, (0, 255, 0), 3)

    # cv.imshow('Marked Rectangular Contours', marked_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return marked_image


if __name__ == '__main__':
    final_images = []
    opening_images = []
    sobel_images = []
    binarized_images = []
    fourier_images = []
    bilateral_images = []
    equalized_images = []

    for i in range(1, 15):
        color_image = cv.imread(INPUT_FOLDER + '/img-' + str(i) + '.png')
        gray_image = cv.imread(INPUT_FOLDER + '/img-' + str(i) + '.png', cv.IMREAD_GRAYSCALE)

        sobel = watanimage.sobel_filter(gray_image)
        sobel = cv.add(sobel, gray_image)
        sobel_images.append(sobel)

        equalized = watanimage.equalize_histogram(gray_image)
        equalized_images.append(equalized)

        opening = watanimage.opening(gray_image)
        opening_images.append(opening)

        bilateral = watanimage.bilateral_filter(gray_image)
        bilateral_images.append(bilateral)

        binarized_img = watanimage.binarize_image(bilateral)
        binarized_images.append(binarized_img)

        final = mark_rectangular_contours(color_image, binarized_img)
        final_images.append(final)

    for bin_img in binarized_images:
        bin_img[bin_img == 1] = 255

    save_images(bilateral_images, 'bilateral-')
    save_images(opening_images, 'opening-')
    save_images(sobel_images, 'sobel-')
    save_images(binarized_images, 'binarized-')
    save_images(final_images, 'final-')
    save_images(equalized_images, 'equalized-')
