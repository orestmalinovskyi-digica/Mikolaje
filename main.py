"""
├── create_test_set.py                # copies files in a controlled random manner into train and test folders
├── detect_and_remove_duplicates.py   # detects and deletes images-duplicates based on their hashes
├── history.p                         # latest model training history dict
├── images/                           # directory with all the images
│   ├── interesting_examples/         # interesting examples like black santa, Grinch etc
│   │   └── exmp/
│   │       └─ *.jpg
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
import shap
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


"""
# An alternative I preferred not to use
def load_test_ds(path="images/test"):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet_v2.preprocess_input)
    ds = lambda: img_gen.flow_from_directory(directory=path,
                                             target_size=(224, 224),
                                             batch_size=2,
                                             class_mode=None)
    ds = tf.data.Dataset.from_generator(ds, output_types=tf.float32)
    return ds
"""


# Plot example pics
def plot_example(images, labels, class_names, n=9, win_name=None):
    # SOURCE: https://www.tensorflow.org/tutorials/load_data/images
    l = round(np.sqrt(n))
    d = np.ceil(n / l).astype(int)

    plt.figure(figsize=(10, 10))
    for i in range(l * d):
        if i == n:
            break
        ax = plt.subplot(l, d, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
        if win_name is not None:
            fig = ax.figure
            fig.canvas.set_window_title(win_name)
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


def visualize_misclassified(ds, model, cl_names, win_name):
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
    plot_example(images, preds, cl_names, len(falses), win_name)  # plot the images and their values


def load_one(path):
    # load the image and preprocess it to fit into classificator
    test_image = load_img(path, target_size=(224, 224))
    image = img_to_array(test_image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    image = tf.convert_to_tensor(image)
    return image


def visualize_one(path, model, cl_names):
    image = load_one(path)
    # predict its label from the model
    pred = np.round(model.predict(image).flatten()).astype(int)

    image = (image * 255. / 2 + 255. / 2)  # rescale the image back, ResNetV2 uses [-1,1] input values
    plot_example(image, pred, cl_names, n=1)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if_load = True      # whether to load a trained model (will train it and save a new model if False)
    if_complex = True   # if we train the model from the beginning, - whether to load simpler or elaborated top layers
    if_load_dss = True  # whether to load training and validation datasets

    if_load_dss = False if not if_load else if_load_dss  # if we train new model, we should load the datasets

    if if_load_dss:
        class_names = pickle.load(open("class_names.p", "rb"))
        train_ds = val_ds = None
    else:
        batch_size = 20
        train, val = load_datasets(batch_size=batch_size)

        # SOURCE: https://www.tensorflow.org/tutorials/load_data/images

        class_names = train.class_names

        for images, labels in train.take(1):
            plot_example(images, labels, class_names)  # plot 9 images with their labels

        # preprocess the images datasets to fit into the model
        train_ds = preprocess(train)
        val_ds = preprocess(val)

        pickle.dump(class_names, open("class_names.p", 'wb'))

    my_model, train_history = get_model(train_dataset=train_ds,
                                        val_dataset=val_ds,
                                        if_load=if_load,
                                        if_complex=if_complex)

    if if_load_dss:
        # load test set and evaluate the model and visualize misclassified images
        test, temp = load_datasets("images/test", 2, 0, (224, 224))
        test = preprocess(test)
        # results = my_model.evaluate(test)
        # print(results)
        visualize_misclassified(test, my_model, class_names, win_name='Misclassified test')

        # load interesting images for this problem
        test, temp = load_datasets("images/interesting_examples", 1, 0, (224, 224))
        test = preprocess(test)
        preds = np.round(my_model.predict(test)).flatten().astype(int)

        # rearrange the dataset so that it can be plottable with plot_examle
        test = list(test)
        test = [x[0] for x in test]  # the 0 element is an image, the 1 element is label
        images = tf.convert_to_tensor(test)[:, 0, :, :, :]
        images = (images * 255. / 2 + 255. / 2)  # rescale the image back, ResNetV2 uses [-1,1] input values
        plot_example(images, preds, class_names, n=images.shape[0])

    else:
        visualize_misclassified(train_ds, my_model, class_names, win_name='Misclassified train')
        visualize_misclassified(val_ds, my_model, class_names, win_name='Misclassified validation')

    # SOURCE: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    print(train_history.keys())
    # summarize history for accuracy
    plt.plot(train_history['acc'])
    plt.plot(train_history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(train_history['loss'])
    plt.plot(train_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    shap_img = load_one("images/train/santa/00000436.jpg")
    e = shap.DeepExplainer(my_model, shap_img)  # 2gi argument to ma być background z iluś trainingowych obrazków
    shap_values = e.shap_values(shap_img)  # tu ma byc lista z ciekawych obrazków
    shap.image_plot(shap_values, -shap_img)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
