import tensorflow as tf
from tensorflow.keras import layers


class DataAugmentation:
    def flip_left_right(X):
        """
        X is expected to be of 4th dimension.
        """
        image_shape = X[0].shape
        flipped_X = np.array([tf.image.flip_left_right(image) for image in X])
        return flipped_X.reshape(len(flipped_X), image_shape[0], image_shape[1], image_shape[2])

    def adjust_brightness(X, brightness_factor):
        """
        X is expected to be of 4th dimension.
        brightness_factor should be between 0 and 1.
        """
        image_shape = X[0].shape
        adjusted_brightness_X = np.array([tf.image.adjust_brightness(image, brightness_factor) for image in X])
        return adjusted_brightness_X.reshape(len(adjusted_brightness_X), image_shape[0], image_shape[1], image_shape[2])

    def rotate_90(X):
        """
        X is expected to be of 4th dimension.
        """
        image_shape = X[0].shape
        rotated_X = np.array([tf.image.rot90(image) for image in X])
        return rotated_X.reshape(len(rotated_X), image_shape[0], image_shape[1], image_shape[2])