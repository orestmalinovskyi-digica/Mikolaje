# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
import pickle
import matplotlib.image as mpimg


# Handling data loading
def load_datasets(direct=None, batch_size=10, val_size=.1, im_shape=(224, 224), seed=8):
    if direct is None:
        direct = "images/train"
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # CAN'T USE VAL SPLIT CAUSE I NEED BOTH VAL AND
        # TEST SPLIT
        directory=direct,
        subset='training',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        color_mode='rgb',
        image_size=im_shape,
        shuffle=True,
        validation_split=val_size,
        seed=seed
    )  # pip install tf-nightly if can't load the class
    # SOURCE:
    # https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test
    """
    DATASET_SIZE = len(list(full_ds))
    v_size = int(val_size * DATASET_SIZE)
    tt_size = int(test_size * DATASET_SIZE)
    tn_size = DATASET_SIZE - v_size - tt_size

    train_ds = full_ds.take(tn_size)  # the ds is already shuffled
    test_ds = full_ds.skip(tn_size)
    val_ds = test_ds.skip(v_size)
    test_ds = test_ds.take(tt_size)

    return train_ds, val_ds.as_numpy_iterator(), test_ds.as_numpy_iterator()  # TODO: probably rewrite the code, it needs to be rescalable and probably test
    # set I will have to create manually"""
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(  # CAN'T USE VAL SPLIT CAUSE I NEED BOTH VAL AND
        # TEST SPLIT
        directory=direct,
        subset='validation',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        color_mode='rgb',
        image_size=im_shape,
        shuffle=True,
        validation_split=val_size,
        seed=seed
    )  # pip install tf-nightly if can't load the class
    return train_ds, val_ds


"""
# An alternative I preferred not to use
def load_ds(path):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    ds = tf.data.Dataset.from_generator(lambda: img_gen.flow_from_directory(path),  # 'images/dataset'),
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=([32, 256, 256, 3], [32, 2])
                                        )
    return ds
"""


# Plot example pics
def plot_example(train, class_names, n=9):
    # SOURCE: https://www.tensorflow.org/tutorials/load_data/images
    l = round(np.sqrt(n))
    d = round(n / l)

    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(l * d):
            ax = plt.subplot(l, d, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()


# normalize dataset according to ResNetV2 needs
def preprocess(dataset):
    # SOURCE: https://www.tensorflow.org/tutorials/load_data/images
    normalized_ds = dataset.map(lambda x, y: (tf.keras.applications.resnet_v2.preprocess_input(x), y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    ds = normalized_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return ds


# Functions of top layers structure

def top_simpler(start_model):
    x = layers.Dense(1, activation="sigmoid")(start_model.output)

    return tf.keras.models.Model(start_model.input, x)


def top_bigger(start_model):
    # SOURCE: https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/
    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(start_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.models.Model(start_model.input, x)


def build_model(if_complex=True):
    # Load the initial model
    init_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    # Plotting model structure to a file
    model_plot_fpath = 'model_plot.png'

    plot_model = os.path.exists(model_plot_fpath)

    if not plot_model:  # if we don't yet have the model plot, make it
        tf.keras.utils.plot_model(init_model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    for layer in init_model.layers:  # deactivate trainability of all layers of the resnet
        layer.trainable = False

    if if_complex:
        model = top_bigger(init_model)  # build several layers on top of the resnet
    else:
        model = top_simpler(init_model)  # put just one dense layer on top

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss='binary_crossentropy', metrics=['acc'])

    return model


def train_and_save(if_complex=True, save_history=True):
    # build the model
    model = build_model(if_complex)

    n_batches = len(list(train_ds))
    resnet_history = model.fit(train_ds, validation_data=val_ds, steps_per_epoch=n_batches, epochs=6)

    model_fname = "saved_model.h5"
    model.save(model_fname)

    if save_history:
        pickle.dump(resnet_history.history, open("history.p", 'wb'))

    return model, resnet_history


def load_model(load_history=True):
    model = tf.keras.models.load_model("saved_model.h5")

    if load_history:
        history = pickle.load(
            open("history.p", "rb")
        )
    else:
        history = None
    return model, history


def get_model(if_load, if_complex):
    if if_load:
        if os.path.exists('saved_model.h5'):
            if os.path.exists('history.p'):
                model, history = load_model()
            else:
                model, history = load_model(load_history=False)
                raise Exception('Could not find the history file')
        else:
            raise NameError('Could not find the model file')
    else:
        # train the model and save it and the history to a file
        model, history = train_and_save(if_complex)
    return model, history


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    batch_size = 20
    train, val = load_datasets(batch_size=batch_size)

    # SOURCE: https://www.tensorflow.org/tutorials/load_data/images

    class_names = train.class_names

    plot_example(train, class_names)  # plot 9 images with their labels

    # preprocess the images datasets to fit into the model
    train_ds = preprocess(train)
    val_ds = preprocess(val)

    """
    image_batch, labels_batch = next(iter(train_ds))  # normalized_ds))
    first_image = image_batch[0]
    # Notice the pixels values are now in `[-1,1]`.
    print(np.min(first_image), np.max(first_image))
    """

    if_load = True  # whether to load a trained model (will train it and save a new model if False)
    if_complex = True  # if we train the model from the beginning, - whether to load simpler or elaborated top layers
    my_model, train_history = get_model(if_load=if_load, if_complex=if_complex)

    test_image = load_img("images/black_santa.jpg", target_size=(224, 224))
    image = img_to_array(test_image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    pred = my_model.predict(image)

    # test_image = mpimg.imread("images/peops/person_252.bmp")
    plt.imshow(test_image)  # .numpy().astype("uint8"))
    plt.title(class_names[round(pred[0][0])])
    plt.axis("off")
    #a = list(val_ds)
    #np.concatenate(list(map(lambda x: x.numpy(), a[:,1])), 0)


    train_preds = my_model.predict(train_ds)
    preds = np.round(train_preds.flatten())

    # transform dataset object to a list:
    train_list = list(train_ds)
    labels = np.array(train_list)[:, 1]

    # Now I need to flatten the labels
    labels = map(lambda x: x.numpy(), labels)
    labels = np.concatenate(list(labels), 0)

    falses = np.argwhere(labels != preds).flatten()
    images = np.array(train_list)[:, 0].flatten()
    #for im in val_ds:
    #    print(im)


    """
    # An alternative
    for i in range(len(list(train_ds))):
        for images, labels in train_ds.take(i+1):  # only take ith element of dataset
            numpy_images = images.numpy()
            numpy_labels = labels.numpy()
            preds = my_model.predict(numpy_images)
            pass
    """
    # TODO: Rename folder dataset to old_dataset, make confusion matrix, show 20-30 images the model didn't predict well
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
