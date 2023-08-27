import numpy as np
import cv2 as cv

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'

# INTERVALO DE BITS
RANGE_MIN = 1  # MIN BITS
RANGE_MAX = 8  # MAX BITS


def quantize_image(image, bits):
    levels = 2 ** bits

    quantized_image = np.floor_divide(image, 256 // levels) * (255 // (levels - 1))

    return quantized_image.astype(np.uint8)


if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE, cv.IMREAD_GRAYSCALE)

    for i in range(RANGE_MIN, RANGE_MAX + 1):
        quantized_image = quantize_image(input_image, i)
        print(f'Quantized image with {i} bits')
        print(quantized_image)

        output_filename = f'{OUTPUT_FOLDER}/quantized_image_{i}.png'
        cv.imwrite(output_filename, quantized_image)
