import numpy as np
import cv2 as cv

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'

# INTERVALO DE AMOSTRAGEM
RANGE_MIN = 2
RANGE_MAX = 16


def sample_image(image, interval):
    height, width, _ = image.shape

    new_height = height // interval
    new_width = width // interval

    sampled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            sampled_image[i, j] = image[i * interval, j * interval]

    return sampled_image


if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE)

    for i in range(RANGE_MIN, RANGE_MAX + 1):
        sampled_image = sample_image(input_image, i)

        output_filename = f'{OUTPUT_FOLDER}/sampled_image_{i}.png'
        cv.imwrite(output_filename, sampled_image)
