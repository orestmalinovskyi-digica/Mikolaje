"""
├── create_test_set.py                # copies files in a controlled random manner into train and test folders
├── detect_and_remove_duplicates.py   # detects and deletes images-duplicates based on their hashes
├── history.p                         # latest model training history dict
├── images/                           # directory with all the images
│   ├── black_santa.jpg               # there were only two black santas in the raw dataset
│   ├── Grinch.jpg                    # Grinch dressed as santa
│   ├── Mrs_Claus.jpg                 # santa's wife otherwise dressed like him
│   ├── Nicolaus.jpg                  # St Nicholas icon
│   ├── old_dataset/                  # deprecated dataset used earlier for training
│   │   ├── person/
│   │   │   └─ *.bmp
│   │   └── santa/
│   │       └─ *.jpg
│   ├── raw/                          # contains all images, that however are already checked for duplicates and
│   │   ├── peops/                    # filtered manually
│   │   │   └─ *.bmp
│   │   └── santas/
│   │       └─ *.jpg
│   ├── test/                         # test directory
│   │   ├── person/
│   │   │   └─ *.bmp
│   │   └── santa/
│   │       └─ *.jpg
│   └── train/                        # train directory
│       ├── person/
│       │   └─ *.bmp
│       └── santa/
│           └─ *.jpg
├── main.py                           # you are here
├── model_plot.png                    # a plot of ResNet50V2 layer structure
├── print_file_structure.py           # a file that helped me make this comment about structure
├── santa_download.py                 # a file that downloads images from a file containing corresponding urls
├── saved_model.h5                    # saved training model
└── urls.txt                          # file containing urls for santa pictures

"""

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
    if val_size:
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
    else:
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # CAN'T USE VAL SPLIT CAUSE I NEED BOTH VAL AND
            # TEST SPLIT
            directory=direct,
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            color_mode='rgb',
            image_size=im_shape,
            shuffle=True,
            seed=seed
        )  # pip install tf-nightly if can't load the class
        return train_ds, None


# An alternative I preferred not to use
def load_test_ds(path="images/test"):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
    ds = lambda: img_gen.flow_from_directory(directory=path,
                                     target_size=(224, 224),
                                     batch_size=1,
                                     class_mode=None)
    ds = tf.data.Dataset.from_generator(ds, output_types=tf.float32)
    return ds


# Plot example pics
def plot_example(images, labels, class_names, n=9):
    # SOURCE: https://www.tensorflow.org/tutorials/load_data/images
    l = round(np.sqrt(n))
    d = round(n / l)

    plt.figure(figsize=(10, 10))
    # for images, labels in ds.take(1):
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


def train_and_save(train_dataset, val_dataset, if_complex=True, save_history=True, ):
    # build the model
    model = build_model(if_complex)

    n_batches = len(list(train_dataset))
    resnet_history = model.fit(train_dataset, validation_data=val_dataset, steps_per_epoch=n_batches, epochs=6)

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


def get_model(if_load, if_complex, train_dataset, val_dataset):
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
        model, history = train_and_save(train_dataset=train_dataset, val_dataset=val_dataset, if_complex=if_complex)
    return model, history


def visualize_misclassified(ds, model, class_names):
    preds = model.predict(ds)
    preds = np.round(preds.flatten()).astype(int)

    # transform dataset object to a list:
    pred_list = list(ds)

    # take labels from the dataset as a numpy array
    labels = np.array(pred_list, dtype=object)[:, 1]

    # Now I need to flatten the labels
    # convert every batch to numpy array
    labels = map(lambda x: x.numpy(), labels)

    # join all batches along 0 axis
    labels = np.concatenate(list(labels), 0)

    # take indexes of where the classificator did a mistake
    falses = np.argwhere(labels != preds).flatten()

    if not len(falses):
        print('\nNo misclassified images!\n')
        return

    # take images from the predicted dataset
    images = np.array(pred_list, dtype=object)[:, 0].flatten()

    # calculate the indexes of the wrongly classified images and their labels
    n_batches = len(list(ds))
    n_labels = labels.shape[0]
    n_cols = np.ceil(n_labels / n_batches)
    row_ns = (falses // n_cols).astype(int)
    column_ns = (falses % n_cols).astype(int)

    # get only wrongly classified images and labels
    images = tf.gather(images[row_ns][0], column_ns)
    images = (images * 255. / 2 + 255. / 2)  # rescale the image back, ResNetV2 uses [-1,1] input values

    preds = preds[falses.astype(int)]  # get wrongly classified images' labels
    plot_example(images, preds, class_names, len(falses))  # plot the images and their values


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if_load = True  # whether to load a trained model (will train it and save a new model if False)
    if_complex = True  # if we train the model from the beginning, - whether to load simpler or elaborated top layers

    if not if_load:  # TODO: in the else clause the train and validation should be None
        batch_size = 20
        train, val = load_datasets(batch_size=batch_size)

        # SOURCE: https://www.tensorflow.org/tutorials/load_data/images

        class_names = train.class_names
        pickle.dump(class_names, open("class_names.p", 'wb'))

        for images, labels in train.take(1):
            plot_example(images, labels, class_names)  # plot 9 images with their labels

        # preprocess the images datasets to fit into the model
        train_ds = preprocess(train)
        val_ds = preprocess(val)

        """
        image_batch, labels_batch = next(iter(train_ds))  # normalized_ds))
        first_image = image_batch[0]
        # Notice the pixels values are now in `[-1,1]`.
        print(np.min(first_image), np.max(first_image))
        """
    else:
        class_names = pickle.load(open("class_names.p", "rb"))
        train_ds = val_ds = None

    my_model, train_history = get_model(train_dataset=train_ds,
                                        val_dataset=val_ds,
                                        if_load=if_load,
                                        if_complex=if_complex)

    """
    test_image = load_img("images/black_santa.jpg", target_size=(224, 224))
    image = img_to_array(test_image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    image = tf.convert_to_tensor(image)
    pred = np.round(my_model.predict(image).flatten()).astype(int)
    image = (image * 255. / 2 + 255. / 2)  # rescale the image back, ResNetV2 uses [-1,1] input values
    plot_example(image, pred, class_names, n=1)
    """

    if if_load:
        # test = load_test_ds()
        test, temp = load_datasets("images/test", 2, 0, (224, 224))
        test = preprocess(test)
        # results = my_model.evaluate(test)
        # print(results)
        visualize_misclassified(test, my_model, class_names)
        test = load_test_ds("images/interesting_examples")
        #x = next(test)
        #image = x[0, :, :, :]
        # preds = np.round(my_model.predict(np.expand_dims(image, axis=0))).flatten().astype(int)
        preds = np.round(my_model.predict(test)).flatten().astype(int)
        plot_example(list(iter(test)), preds, class_names, n=test.n)

    else:
        visualize_misclassified(train_ds, my_model, class_names)
        visualize_misclassified(val_ds, my_model, class_names)

    """
    # An alternative
    for i in range(len(list(train_ds))):
        for images, labels in train_ds.take(i+1):  # only take ith element of dataset
            numpy_images = images.numpy()
            numpy_labels = labels.numpy()
            preds = my_model.predict(numpy_images)
            pass
    """
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
