import tensorflow as tf
import tensorflow_datasets as tfds

def normalize_img(image, label):
    """Normalizes images from `uint8` -> `float32`.
    
    :param image: The image to normalize.
    :param label: The image's label (to ensure the contract).
    """
    return tf.cast(image, tf.float32) / 255, label

def create_model():
    """Load the MNIST dataset, create training/testing pipelines, then
    run the model.
    
    :returns: A tuple containing an image of a digit and a label, and
    dataset info.
    """

    # Load dataset.
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split = ['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ## Create training pipeline
    # Model expects tf.float32, so convert.
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.cache()
    # Apply random transformations and batch elements.
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ## Create evaluation pipeline
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    ## Create and train the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=20,
        validation_data=ds_test,
    )

