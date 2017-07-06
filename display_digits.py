import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
# Import MNIST input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
# Deactivating warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


# Display all digits in a 2D grid
# x_batch: array of N input digits' (flattened) image
# x_width, x_height: dimension of each image to reshape the flattened image
def display_digits(x_batch, x_width, x_height):
    # Check dimensions
    if len(x_batch[0]) != x_width * x_height:
        raise Exception('Dimension for the input image and width/height doesn\'t match')
    num_digits = len(x_batch)
    num_cols = math.floor(math.sqrt(num_digits))
    num_rows = math.ceil(num_digits / num_cols)

    # width of the border between each image
    offset = 1

    # For each row of the grid, append the pixels of each image, row by row
    row_of_ones = [1.0 for _ in range(offset + num_cols * (x_width + offset))]
    disp_grid = [row_of_ones for _ in range(offset)]
    for row_grid in range(num_rows):
        x_row = [list(_) for _ in x_batch[row_grid * num_cols: (row_grid + 1) * num_cols]]
        if row_grid == num_rows - 1 and num_digits < num_rows * num_cols:
            x_row += [list(np.ones(x_width * x_height) * 0.3) for _ in range(num_cols - num_digits % num_cols)]
        # Concatenate the corresponding rows from all images within the same row_grid together
        for row_img in range(x_height):
            disp_row = [1 for _ in range(offset)]
            for x in x_row:
                disp_row += x[row_img * x_height: (row_img + 1) * x_height] + [1 for _ in range(offset)]
            disp_grid.append(disp_row)
        [disp_grid.append(row_of_ones) for _ in range(offset)]
        
    # Print grayscale of all digits
    plt.imshow(disp_grid, cmap=plt.get_cmap('gray_r'))
    plt.show()


def main():
    batch = mnist.test.next_batch(420)
    display_digits(batch[0], IMAGE_SIZE, IMAGE_SIZE)

if __name__ == '__main__':
    main()
