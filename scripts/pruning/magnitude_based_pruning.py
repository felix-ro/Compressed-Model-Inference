# BASED ON https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
import tempfile
import os
import zipfile
import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow import keras


def get_gzipped_model_size(file):
    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file) / (1024 ** 2)  # 1 MiB = 2^20 bytes


def get_lenet_model():
    # LeNet-5 model
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])


def main():
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 and 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = get_lenet_model()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(
      train_images,
      train_labels,
      epochs=6,
      validation_split=0.1,
    )

    _, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)

    keras_file = "scripts/pruning/model.h5"
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    batch_size = 128
    epochs = 2
    validation_split = 0.1

    num_images = train_images.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    pruning_params = {
          'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                   final_sparsity=0.80,
                                                                   begin_step=0,
                                                                   end_step=end_step),
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    model_for_pruning.compile(optimizer='adam',
                              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])

    model_for_pruning.summary()

    logdir = tempfile.mkdtemp()

    callbacks = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(train_images, train_labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_split=validation_split,
                          callbacks=callbacks)

    _, model_for_pruning_accuracy = model_for_pruning.evaluate(
       test_images, test_labels, verbose=0)

    print('Baseline test accuracy:', baseline_model_accuracy)
    print('Pruned test accuracy:', model_for_pruning_accuracy)

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    pruned_keras_file = "scripts/pruning/pruned-model.h5"
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)

    print("Size of gzipped baseline Keras model: %.2f MiB" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned Keras model: %.2f MiB" % (get_gzipped_model_size(pruned_keras_file)))

    model_for_export.compile(optimizer='adam',
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])

    model.evaluate(test_images, test_labels)
    model_for_export.evaluate(test_images, test_labels)


if __name__ == "__main__":
    main()
