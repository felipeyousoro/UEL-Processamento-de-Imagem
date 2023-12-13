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

    # remove contours with approximated area less than 6 and greater than 4
    selected_contours = []

    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)

        if 4 <= len(approx) <= 8:
            selected_contours.append(contour)

    selected_contours = sorted(selected_contours, key=cv.contourArea, reverse=True)[:5]

    for contour in selected_contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return marked_image


if __name__ == '__main__':
    final_images = []
    opening_images = []
    sobel_images = []
    binarized_images = []
    fourier_images = []
    bilateral_images = []
    equalized_images = []
    prewitt_images = []
    gaussian_images = []
    edge_images = []

    for i in range(1, 13):
        color_image = cv.imread(INPUT_FOLDER + '/img (' + str(i) + ').png')
        gray_image = cv.imread(INPUT_FOLDER + '/img (' + str(i) + ').png', cv.IMREAD_GRAYSCALE)

        gaussian = watanimage.gaussian_blur(gray_image)
        gaussian_images.append(gaussian)

        edge = watanimage.sobel_filter(gaussian)
        edge_images.append(edge)

        sobel = cv.addWeighted(gaussian, 0.5, edge, 0.5, 50)
        sobel_images.append(sobel)

        binarized_img = watanimage.binarize_image(sobel)
        binarized_images.append(binarized_img)

        final = mark_rectangular_contours(color_image, binarized_img)
        final_images.append(final)

    for bin_img in binarized_images:
        bin_img[bin_img == 1] = 255

    save_images(prewitt_images, 'prewitt-')
    save_images(gaussian_images, 'gaussian-')
    save_images(bilateral_images, 'bilateral-')
    save_images(opening_images, 'opening-')
    save_images(edge_images, 'edge-')
    save_images(sobel_images, 'sobel-')
    save_images(binarized_images, 'binarized-')
    save_images(final_images, 'final-')
    save_images(equalized_images, 'equalized-')
