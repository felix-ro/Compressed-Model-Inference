# BASED ON https://www.tensorflow.org/lite/performance/post_training_integer_quant
import tensorflow as tf
import numpy as np

from utils import get_gzipped_model_size


def run_tflite_model(model: tf.keras.Sequential, test_images):
    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_images),), dtype=int)
    for i, test_image in enumerate(test_images):
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        test_image = np.expand_dims(test_image, axis=-1)
        interpreter.set_tensor(input_details["index"], test_image)

        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        predictions[i] = output.argmax()

    return predictions


def evaluate_model(model, test_images, test_labels):
    predictions = run_tflite_model(model, test_images=test_images)
    accuracy = (np.sum(test_labels == predictions) * 100) / len(test_images)

    print('Model accuracy is %.4f%% (Number of test samples=%d)' % (accuracy, len(test_images)))


def main():
    model = tf.keras.models.load_model("scripts/pruning/model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    pruned_model = tf.keras.models.load_model("scripts/pruning/pruned-model.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_pruned_quant_model = converter.convert()

    mnist = tf.keras.datasets.mnist
    _, (test_images, test_labels) = mnist.load_data()

    print("Baseline Model:")
    evaluate_model(tflite_quant_model, test_images, test_labels)

    print("Pruned Model:")
    evaluate_model(tflite_pruned_quant_model, test_images, test_labels)

    filename_baseline = "scripts/quantization/model.tflite"
    with open(filename_baseline, 'wb') as f:
        f.write(tflite_quant_model)

    filename_pruned = "scripts/quantization/quant-model.tflite"
    with open(filename_pruned, 'wb') as f:
        f.write(tflite_pruned_quant_model)

    print("Size of gzipped quantized model: %.2f MiB" % (get_gzipped_model_size(filename_baseline)))
    print("Size of gzipped pruned quantized model: %.2f MiB" % (get_gzipped_model_size(filename_pruned)))


if __name__ == "__main__":
    main()
