import numpy as np
import cv2 as cv

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'
RANGE_MIN = 1   # MIN BITS
RANGE_MAX = 8   # MAX BITS


def quantize_image(image, bits):
    # Assumindo o máximo de 256 níveis
    levels = 256 // (2 ** bits)

    # Embora ainda esteja consumindo 1 byte por pixel,
    # a quantidade máxima de bits utilizados sempre será 2^bits.
    quantized_image = (image // levels) * levels


    return quantized_image

if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE)

    for i in range(RANGE_MIN, RANGE_MAX + 1):
        quantized_image = quantize_image(input_image, i)

        output_filename = f'{OUTPUT_FOLDER}/quantized_image_{i}.png'
        cv.imwrite(output_filename, quantized_image)