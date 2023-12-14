import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np

def create_lenet5_model():
    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32, 32, 1), padding="same"))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh'))
    model.add(layers.Flatten())
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

def load_preprocess_mnist(validation_split=0.1):
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    train_images = tf.pad(train_images.astype('float32'), [[0, 0], [2,2], [2,2]]) / 255.0
    test_images = tf.pad(test_images.astype('float32'), [[0, 0], [2,2], [2,2]]) / 255.0
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]

    total_train = train_images.shape[0]
    val_size = int(total_train * validation_split)
    train_data = (train_images[val_size:], train_labels[val_size:])
    val_data = (train_images[:val_size], train_labels[:val_size])

    return train_data, val_data, (test_images, test_labels)

model = create_lenet5_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_preprocess_mnist()

history = model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(val_images, val_labels))

ls = [layer for layer in model.layers]

filters1, biases1 = ls[0].get_weights()
filters2, biases2 = ls[2].get_weights()
filters3, biases3 = ls[4].get_weights()
dense1, dense_biases1 = ls[6].get_weights()[0].transpose(), ls[6].get_weights()[1]
dense2, dense_biases2 = ls[7].get_weights()[0].transpose(), ls[7].get_weights()[1]

np.savetxt("../weights/filters_layer_0.txt", filters1.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/biases_layer_0.txt", biases1.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/filters_layer_2.txt", filters2.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/biases_layer_2.txt", biases2.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/filters_layer_4.txt", filters3.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/biases_layer_4.txt", biases3.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/dense_layer_6.txt", dense1.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/dense_biases_layer_6.txt", dense_biases1.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/dense_layer_7.txt", dense2.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/dense_biases_layer_7.txt", dense_biases2.flatten(), delimiter=',', fmt='%f')
np.savetxt("../weights/test.txt", np.array(test_images[0]).flatten(), delimiter=',', fmt='%f')
