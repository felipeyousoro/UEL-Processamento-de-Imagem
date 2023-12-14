import math as m
import cv2 as cv
import numpy as np
import pytesseract as pytesseract

import watanimage

# PATHING DAS IMAGES
INPUT_FOLDER = 'inputs'
OUTPUT_FOLDER = 'outputs'


def save_images(imgs, name):
    for i in range(len(imgs)):
        cv.imwrite(OUTPUT_FOLDER + '/' + name + str(i + 1) + '.png', imgs[i])


def rectangular_contours_to_image(binarized_img: np.ndarray, contour: np.ndarray, height: int,
                                  width: int) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.uint8)

    cv.drawContours(image, [contour], -1, 1, thickness=cv.FILLED)

    x, y, w, h = cv.boundingRect(contour)

    cropped_image = image[y:y + h, x:x + w]

    result_image = np.where(cropped_image == 1, binarized_img[y:y + h, x:x + w], 0)

    return result_image


def mark_rectangular_contours(colored_image: np.ndarray, binarized_image: np.ndarray, countours) -> np.ndarray:
    marked_image = colored_image.copy()

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return marked_image


if __name__ == '__main__':
    NO_IMAGES = 13

    for i in range(1, NO_IMAGES + 1):
        images = []
        print(i)
        color_image = cv.imread(INPUT_FOLDER + '/img (' + str(i) + ').png')
        gray_image = cv.imread(INPUT_FOLDER + '/img (' + str(i) + ').png', cv.IMREAD_GRAYSCALE)

        # Aplica o pré-processamento
        gaussian = watanimage.gaussian_blur(gray_image)
        edge = watanimage.sobel_filter(gaussian)
        sobel = cv.addWeighted(gaussian, 0.7, edge, 0.3, 20)
        binarized_img = watanimage.binarize_image(sobel)

        # Busca os contornos
        contours, hierarchy = cv.findContours(binarized_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        selected_contours = []
        for contour in contours:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02 * peri, True)

            # Utilizar len = 4 já funciona para a maioria dos casos
            # mas algumas placas (especialamente do Mercusul)
            # não são reconhecidas corretamente, e, na base dos testes,
            # ter um "limite superior" de 7 vértices funciona bem
            if 4 <= len(approx) <= 7:
                selected_contours.append(contour)

        # Sinceramente, isso aqui é desnescessário,
        # mas, se não fizer, a imagem final fica com
        # muitos contornos pequenos marcados, deixando
        # muito poluído, e me poupa de esperar mais
        # tempo quando for processar os textos
        selected_contours = sorted(selected_contours, key=cv.contourArea, reverse=True)[:5]

        # Marcando contornos para visualizar
        final = mark_rectangular_contours(color_image, binarized_img, selected_contours)
        images.append(final)

        cropped_contours = []
        for contour in selected_contours:
            contour_img = rectangular_contours_to_image(binarized_img, contour, binarized_img.shape[0],
                                                        binarized_img.shape[1])
            contour_img[contour_img == 1] = 255
            cropped_contours.append(contour_img)

        binarized_img[binarized_img == 1] = 255
        images.append(binarized_img)

        # Aplica o Tesseract para detectar o texto
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        for cropped_contour in cropped_contours:
            # Motivo do PSM
            # https://stackoverflow.com/a/44632770
            text = pytesseract.image_to_string(cropped_contour, config='--psm 7')
            images.append(cropped_contour)
            # Removendo espaços e caracteres especiais
            text = ''.join(e for e in text if e.isalnum())
            if len(text) > 5:
                print(text)

        save_images(images, 'img (' + str(i) + ')-')
