import cv2 as cv
import numpy as np
from colors import Colors

# PATHING DAS IMAGES
INPUT_IMAGE = 'akari.png'
OUTPUT_FOLDER = 'outputs'


def get_neighbors(image, x, y):
    height, width, _ = image.shape

    neighbors = []

    if x > 0:
        neighbors.append([x - 1, y])
    if x < height - 1:
        neighbors.append([x + 1, y])
    if y > 0:
        neighbors.append([x, y - 1])
    if y < width - 1:
        neighbors.append([x, y + 1])

    return neighbors


def label_components_dfs(image, x, y, visited, component):
    if visited[x][y] == 1:
        return

    component.append([x, y])
    visited[x][y] = 1
    neighbors = get_neighbors(image, x, y)

    for neighbor in neighbors:
        if (image[neighbor[0]][neighbor[1]] == np.array([1, 1, 1])).all():
            label_components_dfs(image, neighbor[0], neighbor[1], visited, component)

    return


def label_components(image):
    components = []

    height, width, _ = image.shape
    visited = np.zeros((height, width), int)

    for x in range(height):
        for y in range(width):
            if visited[x][y] == 1:
                continue
            if (image[x][y] == np.array([1, 1, 1])).all():
                component = []
                label_components_dfs(image, x, y, visited, component)

                components.append(component)
                continue

    return components


def invert_binary_image(image):
    for x in range(len(image)):
        for y in range(len(image[x])):
            if (image[x][y] == np.array([1, 1, 1])).all():
                image[x][y] = 0
            else:
                image[x][y] = 1

    return image


if __name__ == '__main__':
    input_image = cv.imread(INPUT_IMAGE)
    _, binary_image = cv.threshold(input_image, 127, 1, cv.THRESH_BINARY)
    # Inverter para poder marcar melhores componentes,
    # a imagem original tava usando o branco para o
    # fundo
    binary_image = invert_binary_image(binary_image)

    labeled_components = label_components(binary_image)
    binary_image = binary_image * 255

    current_color = 0
    for component in labeled_components:
        for pixel in component:
            binary_image[pixel[0]][pixel[1]] = Colors[current_color].value
        current_color = (current_color + 1) % len(Colors)

    cv.imwrite(f'{OUTPUT_FOLDER}/componentized_image.png', binary_image)
