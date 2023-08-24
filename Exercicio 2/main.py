import numpy as np
import cv2 as cv

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'
RANGE_MIN = 2
RANGE_MAX = 16

def quantize_image(image, levels):
    # 256 = Number of bits in a pixel
    quantization_interval = 256 // levels

    print(f'Quantization interval: {quantization_interval}')
    print(f'{(image[0][0])}')
    print(f'{(image[0][0] // quantization_interval) * quantization_interval}')

    quantized_image = (image // quantization_interval) * quantization_interval

    return quantized_image

if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE)

    for i in range(RANGE_MIN, RANGE_MAX + 1):
        quantized_image = quantize_image(input_image, i)

        output_filename = f'{OUTPUT_FOLDER}/quantized_akari_{i}.png'
        cv.imwrite(output_filename, quantized_image)