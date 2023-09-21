import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'

CONTRAST = 1
BRIGHTNESS = 40


def build_histogram(image: np.ndarray) -> np.ndarray:
    histogram = np.zeros(256, int)

    height, width = image.shape

    for x in range(height):
        for y in range(width):
            histogram[image[x][y]] += 1

    return histogram


def plot_histogram(image: np.ndarray, output: str = None, text: str = None):
    histogram = build_histogram(image)

    plt.figure(figsize=(10, 5))
    plt.bar(range(256), histogram, color='gray')
    plt.xlabel('Intensidade')
    plt.ylabel('Quantidade de Pixels')
    plt.title(text)
    plt.savefig(f'{OUTPUT_FOLDER}/{output}.png')


def transform_image(image: np.ndarray, contrast: float, brightness: int) -> np.ndarray:
    height, width = image.shape

    for x in range(height):
        for y in range(width):
            new_value = contrast * image[x][y] + brightness
            if new_value > 255:
                new_value = 255
            image[x][y] = new_value

    return image


def plot_transform_function(output: str, text: str, contrast: float, brightness: int):
    x = np.arange(256)
    y = contrast * x + brightness
    for i in range(256):
        y[i] = int(y[i])
        if y[i] > 255:
            y[i] = 255

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='gray')
    plt.xlabel('Níveis de cinza')
    plt.ylabel('Intensidade')
    plt.title(text)
    plt.savefig(f'{OUTPUT_FOLDER}/{output}.png')


if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE, cv.IMREAD_GRAYSCALE)

    plot_histogram(input_image, 'original_hist', 'Histograma da Imagem Original')

    transformed_image = transform_image(input_image, CONTRAST, BRIGHTNESS)
    plot_transform_function('transform_function', 'Função de Transformação', CONTRAST, BRIGHTNESS)

    plot_histogram(transformed_image, 'transformed_hist', 'Histograma da Imagem Transformada')
    cv.imwrite(f'{OUTPUT_FOLDER}/transformed_image.png', transformed_image)
