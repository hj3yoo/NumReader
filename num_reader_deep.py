import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# NumReader - OCR for handwritten numeric using Tensorflow & MNIST
#

FLAGS = None


# Tensor initialization functions
# Source: https://www.tensorflow.org/get_started/mnist/pros
def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolutional_network_model(x):
    # Modelling convolutional neural network
    # Input layer: 28 x 28 x 1 tensor
    # 1st hidden layer: 14 x 14 x 32 tensor (pooled to half resolution, depth increased to 32)
    # 2nd hidden layer: 7 x 7 x 64 tensor (pooled to half resolution, depth doubled)
    # Fully-connected layer: 1 x 1 x 1024 tensor (combine the previous feature map into 2^10 features vector)
    # Output layer: 1 x 1 x 10 tensor (mapping 2^10 features to 10 output classes)
    # For neurons, use Rectified Linear Unit (ReLU): f(x) ~ max(0,x) - Computationally cheaper than sigmoid

    # Input layer
    # -1 on the first dimension means it can accept any number of input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 1st hidden layer
    # Patch size of the convolution is 5
    # Mapping to 32 features (=depth)
    # Pool to half resolution of 14x14
    w_conv_1 = weight_variable([5, 5, 1, 32])
    b_conv_1 = bias_variable([32])
    h_conv_1 = tf.nn.relu(conv2d(x_image, w_conv_1) + b_conv_1)
    h_pool_1 = max_pool_2x2(h_conv_1)

    # 2nd hidden layer
    # double the number of features, and pool to half resolution
    w_conv_2 = weight_variable([5, 5, 32, 64])
    b_conv_2 = bias_variable([64])
    h_conv_2 = tf.nn.relu(conv2d(h_pool_1, w_conv_2) + b_conv_2)
    h_pool_2 = max_pool_2x2(h_conv_2)

    # Fully-connected layer
    # From 7 x 7 x 64 tensor, extract 1024 (2^10) features
    h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])  # flattening previous layer's activation tensor
    w_fc_1 = weight_variable([7 * 7 * 64, 1024])
    b_fc_1 = bias_variable([1024])
    h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, w_fc_1) + b_fc_1)

    # Dropout: controls the complexity of the model, prevents co-adaptation of features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob)

    # Output layer
    # Map the 1024 features to 10 output classes, one for each digit
    w_fc_2 = weight_variable([1024, 10])
    b_fc_2 = bias_variable([10])

    y_approx = tf.matmul(h_fc_1_drop, w_fc_2) + b_fc_2
    return y_approx, keep_prob


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Placeholder for input data & label
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # Build the neural network
    y_approx, keep_prob = convolutional_network_model(x)

    # Define the loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_approx))

    # Attempt with Adam Optimizer (https://arxiv.org/pdf/1412.6980v8.pdf): need further research
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Training using gradient descent (lambda = 0.015)
    train_step = tf.train.GradientDescentOptimizer(0.015).minimize(cross_entropy)

    # Prediction is accurate if the output tensor is a one-hot coding of the correct output class
    correct_prediction = tf.equal(tf.argmax(y_approx, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        # Incremental gradient descent: for each iteration, pull out small random set of input to train the parameters
        # Since the calculation of loss function requires summation over the entire input set, using a small batch of
        # arbitrary inputs (such that enough number of batches sufficiently approximate the whole sample) results in
        # faster, and more responsive adjustment in smaller step.
        for i in range(10000):
            train_batch = mnist.train.next_batch(50)
            # For every 100 iterations (=steps), test the training accuracy
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: train_batch[0], y: train_batch[1], keep_prob: 1.0})
                print('Training accuracy after iteration #%d: %g%%' % (i, train_accuracy * 100))
            train_step.run(feed_dict={x: train_batch[0], y: train_batch[1], keep_prob: 0.5})

            # For every 1000 iterations, test the prediction accuracy
            if i % 1000 == 0:
                test_batch = mnist.test.next_batch(1000)
                test_accuracy = accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
                print('Test accuracy after iteration #%d: %g%%' % (i, test_accuracy * 100))

    # When the training is done, conduct the final prediction test
    final_test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
    print('Final test accuracy: %g%%' % final_test_accuracy * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Parser initialization
    # Source: https://www.tensorflow.org/get_started/mnist/pros
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
