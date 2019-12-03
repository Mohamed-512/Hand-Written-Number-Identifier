import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.lite as lite


# Downloading MNIST data set from keras
mnist = tf.keras.datasets.mnist
(image_data_training, label_data_training), (image_data_testing, label_data_testing) = mnist.load_data()
image_data_training, image_data_testing = image_data_training / 255.0, image_data_testing / 255.0
image_data_training, image_data_testing = np.expand_dims(image_data_training, axis=-1), np.expand_dims(
    image_data_testing, axis=-1)


def get_balanced_data(count_per_class):
    current_counts = [0]*10
    images, labels = [], []

    for i in range(len(label_data_training)):
        label = label_data_training[i]
        if current_counts[label] >= count_per_class:
            continue
        current_counts[label] += 1
        labels.append(label)
        images.append(image_data_training[i])

    return images, labels


def train(imgs, lbls, epochs=1):
    """
    Creates a model and train it using the training MNIST set then saves it in a '.h5' file to be reused
    :type imgs: images to train on
    :type lbls: labels of those images
    :return: None
    """

    # Input -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> flatten -> dense layer
    inputs = layers.Input(shape=(28, 28, 1))
    c1 = layers.Conv2D(35, (5, 5), padding="valid", activation=tf.nn.relu)(inputs)
    m1 = layers.MaxPool2D((2, 2), (2, 2))(c1)
    c2 = layers.Conv2D(35, (4, 4), padding="valid", activation=tf.nn.relu)(m1)
    m2 = layers.MaxPool2D((3, 3), (3, 3))(c2)
    f = layers.Flatten()(m2)
    outputs = layers.Dense(10, activation=tf.nn.softmax)(f)

    # Structuring the model
    model = models.Model(inputs, outputs)

    # Printing the model's summary
    model.summary()

    # Optimizer and loss function specifying
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # The commented part below was used for debugging and generating '.h5' after each 10k new images seen
    # for i in range(7):
    #     model.fit(image_data_training[i*10000:(i+1)*10000], label_data_training[i*10000:(i+1)*10000], epochs=1)
    #     f_name = 'mnist_model' + str((i+1)*10) + 'k.h5'
    #     model.save(f_name)  # creates a HDF5 file 'my_model.h5'

    # Training the model and saving it in '.h5' for web/RPi camera and '.tflite' for possible Android app
    model.fit(imgs, lbls, epochs=epochs)
    model.save('mnist_model_epoch12_5000imgs.h5')  # creates a HDF5 file 'my_model.h5'
    converter = lite.TocoConverter.from_keras_model_file("mnist_model.h5")
    tflite_model = converter.convert()
    open("mnist_model.tflite", "wb").write(tflite_model)


def test_all(model_name):
    """
    Test a given model on the whole MNIST training set
    * Used for debugging
    :return: None
    """
    model = models.load_model(model_name)
    test_loss, test_acc = model.evaluate(image_data_testing, label_data_testing)
    print("Test Loss: {0} - Test Acc: {1}".format(test_loss, test_acc))


def test(image, model_name):
    """
    Test a given model on single 2D image
    * Used for debugging
    :return: None
    """
    model = models.load_model(model_name)
    prediction = model.predict(np.array([image]))
    return prediction


# train(image_data_training, label_data_training)

test_all('mnist_model_epoch12_5000imgs.h5')

# imgs, lbls = get_balanced_data(500)
# imgs, lbls = np.array(imgs), np.array(lbls)
# print(np.array(lbls).shape, np.array(imgs).shape)
# print(imgs[0].shape)
# train(imgs, lbls, 12)
