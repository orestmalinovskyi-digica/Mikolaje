# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf


def load_datasets(direct=None, batch_size=10, val_size=.1, test_size=.1, im_shape=(224, 224)):
    if direct is None:
        direct = "images/dataset"
    full_ds = tf.keras.preprocessing.image_dataset_from_directory(  # CAN'T USE VAL SPLIT CAUSE I NEED BOTH VAL AND
        # TEST SPLIT
        directory=direct,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        color_mode='rgb',
        image_size=im_shape,
        shuffle=True)  # pip install tf-nightly
    # SOURCE:
    # https://stackoverflow.com/questions/48213766/split-a-dataset-created-by-tensorflow-dataset-api-in-to-train-and-test
    DATASET_SIZE = len(list(full_ds))
    v_size = int(val_size * DATASET_SIZE)
    tt_size = int(test_size * DATASET_SIZE)
    tn_size = DATASET_SIZE - v_size - tt_size

    train_ds = full_ds.take(tn_size)  # the ds is already shuffled
    test_ds = full_ds.skip(tn_size)
    val_ds = test_ds.skip(v_size)
    test_ds = test_ds.take(tt_size)

    return train_ds, val_ds, test_ds # TODO: probably rewrite the code, it needs to be rescalable and probably test
    # set I will have to create manually


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    train, val, test = load_datasets()

    model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    train = tf.keras.applications.resnet_v2.preprocess_input(train)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
train, val, test = load_datasets()
train = tf.keras.applications.resnet_v2.preprocess_input(train)