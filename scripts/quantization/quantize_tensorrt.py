import tensorflow as tf
import numpy as np
from tensorflow_quantization.quantize import quantize_model
from tensorflow_quantization.custom_qdq_cases import ResNetV1QDQCase
from tensorflow_quantization.utils import convert_saved_model_to_onnx


def compile_model(model, lr=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def main():
    num_samples = 100
    input_shape = (224, 224, 3)
    random_input_data = np.random.rand(num_samples, *input_shape)
    random_labels = np.random.randint(0, 1000, size=num_samples)

    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights="imagenet",
        classes=1000,
        classifier_activation="softmax",
    )

    compile_model(model)
    _, baseline_model_accuracy = model.evaluate(random_input_data, random_labels)
    print("Baseline val accuracy: {:.3f}%".format(baseline_model_accuracy * 100))

    model_save_path = "scripts/quantization/model_baseline"
    model.save(model_save_path)
    convert_saved_model_to_onnx(saved_model_dir=model_save_path,
                                onnx_model_path=model_save_path + ".onnx")

    q_model = quantize_model(model, custom_qdq_cases=[ResNetV1QDQCase()])

    compile_model(q_model)
    _, qat_model_accuracy = q_model.evaluate(random_input_data, random_labels)
    print("QAT val accuracy: {:.3f}%".format(qat_model_accuracy*100))

    model_save_path = "scripts/quantization/model_qat"
    model.save(model_save_path)
    convert_saved_model_to_onnx(saved_model_dir=model_save_path,
                                onnx_model_path=model_save_path + ".onnx")


if __name__ == "__main__":
    main()
