import os
import csv
import cv2
import numpy as np
import sklearn

from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Model(object):

    def __init__(self):
        self.model = None

    def build(self,
              input_shape=(160, 320, 3),
              learning_rate=0.0001,
              dropout_prob=0.7):

        model = Sequential()

        # Normalize image
        model.add(Lambda(lambda x: (x / 127.5) - 1,
                         input_shape=input_shape,
                         output_shape=input_shape))

        # Crop input images for less processing.
        model.add(Cropping2D(cropping=((50, 20), (0, 0)),
                             #input_shape=input_shape
                             ))
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
        model.add(Dense(100, activation=activation))
        model.add(Dense(50, activation=activation))
        model.add(Dense(1, activation=activation))

        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        self.model = model
        return model

    def save(self, file_name):
        self.model.save(file_name)

    def load(self, file_name):
        self.model = load_model(file_name)

    def __flip(self, image, angle):
        return np.fliplr(image), - angle

    def __convert_to_rgb(self, image):
        res = image.astype(np.uint8)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        return res

    def __samples_generator(self, samples, batch_size=32):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    #name = './IMG/' + batch_sample[0].split('/')[-1]
                    name = batch_sample[0]

                    center_image = self.__convert_to_rgb(cv2.imread(name))
                    center_angle = float(batch_sample[3])

                    images.append(center_image)
                    angles.append(center_angle)

                    flipped_image, flipped_angle = self.__flip(center_image, center_angle)
                    images.append(flipped_image)
                    angles.append(flipped_angle)

                    # recovery
                    correction = 0.2
                    left_angle = center_angle + correction
                    left_image = self.__convert_to_rgb(cv2.imread(batch_sample[1]))
                    images.append(left_image)
                    angles.append(left_angle)

                    flipped_image, flipped_angle = self.__flip(left_image, left_angle)
                    images.append(flipped_image)
                    angles.append(flipped_angle)

                    right_angle = center_angle - correction
                    right_image = self.__convert_to_rgb(cv2.imread(batch_sample[2]))
                    images.append(right_image)
                    angles.append(right_angle)

                    flipped_image, flipped_angle = self.__flip(right_image, right_angle)
                    images.append(flipped_image)
                    angles.append(flipped_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def train(self, csv_db, nb_epoch=10):
        samples = []
        with open(csv_db) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)

        train_generator = self.__samples_generator(train_samples, batch_size=32)
        validation_generator = self.__samples_generator(validation_samples, batch_size=32)

        self.model.fit_generator(train_generator,
                                 samples_per_epoch=len(train_samples),
                                 validation_data=validation_generator,
                                 nb_val_samples=len(validation_samples),
                                 nb_epoch=nb_epoch)

if __name__ == "__main__":
    m = Model()
    #m.build()
    m.load("model.h5")
    # m.train("../p3_data/t1_middle/driving_log.csv")
    # m.train("../p3_data/t1_back/driving_log.csv")
    # m.train("../p3_data/t1_recover/driving_log.csv")
    # m.train("../p3_data/t1_recover2/driving_log.csv")
    # m.train("../p3_data/t1_recover3/driving_log.csv")
    # m.train("../p3_data/t1_recover4/driving_log.csv")
    #m.train("../p3_data/t1_recover5/driving_log.csv")
    m.train("../p3_data/t1_recover6/driving_log.csv")
    m.save("model.h5")
