import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'

CONTRAST = 1.01
BRIGHTNESS = 0
GRAY_LEVELS = 256


def build_histogram(image: np.ndarray) -> np.ndarray:
    histogram = np.zeros(GRAY_LEVELS, int)

    height, width = image.shape

    for x in range(height):
        for y in range(width):
            histogram[image[x][y]] += 1

    return histogram


def plot_histogram(image: np.ndarray, output: str = None, text: str = None):
    histogram = build_histogram(image)

    plt.figure(figsize=(10, 5))
    plt.bar(range(GRAY_LEVELS), histogram, color='gray')
    plt.xlabel('Intensidade')
    plt.ylabel('Quantidade de Pixels')
    plt.title(text)
    plt.savefig(f'{OUTPUT_FOLDER}/{output}.png')


# def transform_image(image: np.ndarray, contrast: float, brightness: int) -> np.ndarray:
#     height, width = image.shape
#
#     for x in range(height):
#         for y in range(width):
#             new_value = contrast * image[x][y] + brightness
#             if new_value > GRAY_LEVELS - 1:
#                 new_value = GRAY_LEVELS - 1
#             image[x][y] = new_value
#
#     return image
#
#
# def plot_transform_function(output: str, text: str, contrast: float, brightness: int):
#     x = np.arange(GRAY_LEVELS)
#     y = contrast * x + brightness
#     for i in range(GRAY_LEVELS):
#         y[i] = int(y[i])
#         if y[i] > GRAY_LEVELS - 1:
#             y[i] = GRAY_LEVELS - 1
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(x, y, color='gray')
#     plt.xlabel('Níveis de cinza')
#     plt.ylabel('Intensidade')
#     plt.title(text)
#     plt.savefig(f'{OUTPUT_FOLDER}/{output}.png')


def build_transform_map(image: np.ndarray) -> np.ndarray:
    histogram = build_histogram(image)
    height, width = image.shape

    p_values = (histogram / (height * width))

    s_values = np.zeros(GRAY_LEVELS, float)

    for i in range(GRAY_LEVELS):
        s_values[i] = sum(p_values[:i + 1])

    transform_map = np.zeros(GRAY_LEVELS, int)
    for i in range(GRAY_LEVELS):
        transform_map[i] = int(s_values[i] * (GRAY_LEVELS - 1))

    return transform_map


def plot_transform_map(transform_map: np.ndarray, output: str, text: str):
    x = np.arange(GRAY_LEVELS)

    plt.figure(figsize=(10, 5))
    plt.plot(x, transform_map, color='gray')
    plt.xlabel('Níveis de cinza')
    plt.ylabel('Intensidade')
    plt.title(text)
    plt.savefig(f'{OUTPUT_FOLDER}/{output}.png')

def transform_image(image: np.ndarray, transform_map: np.ndarray) -> np.ndarray:
    height, width = image.shape

    for x in range(height):
        for y in range(width):
            image[x][y] = transform_map[image[x][y]]

    return image

if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE, cv.IMREAD_GRAYSCALE)

    plot_histogram(input_image, 'original_hist', 'Histograma da Imagem Original')

    transform_map = build_transform_map(input_image)
    plot_transform_map(transform_map, 'transform_map', 'Função de Transformação')

    transformed_image = transform_image(input_image, transform_map)
    plot_histogram(transformed_image, 'transformed_hist', 'Histograma da Imagem Transformada')

    cv.imwrite(f'{OUTPUT_FOLDER}/transformed_image.png', transformed_image)