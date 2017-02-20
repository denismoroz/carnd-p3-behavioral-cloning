import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential
from keras.layers import Convolution2D, Cropping2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Model(object):

    def __init__(self):
        self.model = None

    """
        CNN based on NVIDIA Prototype based on http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    def build(self,
              input_shape=(160, 320, 3),  # This shape provided by simulator
              learning_rate=0.0001,
              dropout_prob=0.2):

        model = Sequential()

        # Crop Image
        model.add(Cropping2D(cropping=((75, 20),   # cropping top and bottom
                                       (0, 0)),  # cropping left and right
                             input_shape=input_shape))
        # Normalize Image
        model.add(Lambda(lambda x: (x / 127.5) - 1))  # Elu activation allows to use negative values.

        # Crop input images for less processing.
        activation = "elu"

        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation=activation))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))

        model.add(Flatten())

        model.add(Dropout(dropout_prob))
        model.add(Dense(1164, activation=activation))

        model.add(Dropout(dropout_prob))
        model.add(Dense(800, activation=activation))

        model.add(Dropout(dropout_prob))
        model.add(Dense(400, activation=activation))

        model.add(Dropout(dropout_prob))
        model.add(Dense(100, activation=activation))
        model.add(Dense(1, activation=activation))

        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        self.model = model
        return model

    def save(self, file_name):
        self.model.save(file_name)

    def load(self, file_name):
        self.model = load_model(file_name)

    """ Flip image and return flipped steering wheel angle. """
    def __flip(self, image, angle):
        return np.fliplr(image), - angle

    """
        Change brightness and convert image to RGB.
        Used code:
            https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.gw9hfaxvd
    """
    @staticmethod
    def preprocess_frame(image):
        res = image.astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
        random_bright = .25 + np.random.uniform()
        res[:, :, 2] = res[:, :, 2] * random_bright
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        return res

    """
        Load single frame.
        Run pre-processing pipeline on it.
        Flip it.
    """
    def __process_single_frame(self, frame_file_name, angle):
        frames = []
        angles = []

        image = cv2.imread(frame_file_name)

        if image is None:
            print("Failed to read image {0}".format(frame_file_name))
            return frames, angles

        frame = self.preprocess_frame(image)

        frames.append(frame)
        angles.append(angle)

        flipped_frame, flipped_angle = self.__flip(frame, angle)

        frames.append(flipped_frame)
        angles.append(flipped_angle)

        return frames, angles

    """
        Input: single record from csv file:
        Output: 6 frames and 6 angles:
            1 - central frame image + angle
            2 - flipped central frame + flipped angle

            3 - left frame image + angle
            4 - flipped left frame + flipped angle

            5 - right frame image + angle
            6 - flipped right frame + flipped angle
    """
    def __process_csv_record(self, record):

        frames = []
        angles = []

        central_image_file_name = record[0]
        center_angle = float(record[3])

        f, a = self.__process_single_frame(central_image_file_name, center_angle)
        frames += f
        angles += a

        # recovery delta for left and right images
        correction = 0.2

        left_angle = center_angle + correction
        f, a = self.__process_single_frame(record[1], left_angle)
        frames += f
        angles += a

        right_angle = center_angle - correction
        f, a = self.__process_single_frame(record[2], right_angle)
        frames += f
        angles += a

        return frames, angles

    """
        Generator to load features by portions.
    """
    def __samples_generator(self, samples, batch_size=32):
        num_samples = len(samples)
        batch_size = int(batch_size / 6)

        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    f, a = self.__process_csv_record(batch_sample)
                    images += f
                    angles += a

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    """
        Main entrance point for training.

        Input:
            list of csv files with links to training data set.

        It loads all csv files, split training set and validation set.
        Creates training set generator

    """
    def train(self, csv_dbs, nb_epoch=3):
        samples = []
        for csv_db in csv_dbs:
            with open(csv_db) as csv_file:
                reader = csv.reader(csv_file)
                for line in reader:
                    samples.append(line)

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        train_generator = self.__samples_generator(train_samples, batch_size=36)
        validation_generator = self.__samples_generator(validation_samples, batch_size=36)

        self.model.fit_generator(train_generator,
                                 samples_per_epoch=len(train_samples) * 6,
                                 validation_data=validation_generator,
                                 nb_val_samples=len(validation_samples) * 6,
                                 nb_epoch=nb_epoch)

if __name__ == "__main__":
    m = Model()
    m.build()

    # First version of working data set
    # m.train("../p3_data/t1_middle/driving_log.csv")
    # m.train("../p3_data/t1_back/driving_log.csv")
    # m.train("../p3_data/t1_recover/driving_log.csv")
    # m.train("../p3_data/t1_recover2/driving_log.csv")
    # m.train("../p3_data/t1_recover3/driving_log.csv")
    # m.train("../p3_data/t1_recover4/driving_log.csv")
    # m.train("../p3_data/t1_recover5/driving_log.csv")
    # m.train("../p3_data/t1_recover6/driving_log.csv")
    # end of first version of working data set

    # First Track features set
    t1_features_dbs = [
         "../p3_data/t12_middle/driving_log.csv",
         "../p3_data/t12_back/driving_log.csv",
         "../p3_data/t12_recover/driving_log.csv",
    ]

    # Second Track features set
    t2_features_dbs = [
        "../p3_data/t2_middle/driving_log.csv",
        "../p3_data/t2_back/driving_log.csv",
        "../p3_data/t2_recover/driving_log.csv",
        "../p3_data/t2_recover7/driving_log.csv",
        "../p3_data/t2_recover9/driving_log.csv",
        "../p3_data/t2_recover10/driving_log.csv",
    ]

    m.train(t1_features_dbs + t2_features_dbs)

    m.save("model.h5")
