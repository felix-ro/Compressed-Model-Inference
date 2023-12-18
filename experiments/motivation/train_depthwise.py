import tensorflow as tf

from utils import getDataset

def residual(x, filters, kernel_size=3, strides=1, activation="relu"):
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding="SAME")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding="SAME")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=strides, padding="SAME")(x)

    if strides != 1 or shortcut.shape[-1] != filters:
      shortcut = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides, padding="SAME")(shortcut)
      shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation(activation)(x)

    return x

def create_model(input_layer, num_classes):
    x = tf.keras.layers.Conv2D(32, kernel_size=4, strides=2, padding="SAME")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residual(x, filters=64, strides=1)
    x = residual(x, filters=64, strides=1)
    x = residual(x, filters=64, strides=1)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding="SAME")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=1, strides=1, padding="SAME")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

def main():
    dataset = getDataset()
    batch_size = 128
    train_data = dataset.training_dataset().batch(batch_size).prefetch(1) 
    valid_data = dataset.validation_dataset().batch(batch_size).prefetch(1)

    input_shape = dataset.sample_shape()
    num_classes = dataset.label_count()
    input_layer = tf.keras.layers.Input(shape=input_shape)

    model = create_model(input_layer=input_layer, num_classes=num_classes)

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.003),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(train_data, validation_data=valid_data, epochs=35, callbacks=[early_stopping])

    model.save("model-depthwise.h5")

    test_data = dataset.testing_dataset().batch(64)
    model.evaluate(test_data)

if __name__ == "__main__":
    main()