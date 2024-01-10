import tensorflow as tf
import numpy as np
from timeit import Timer

from utils import get_gzipped_model_size

# #################### CONFIGURE BEFORE RUNNING #####################
# DEVICE_NAME = "P100"
DEVICE_NAME = "A100"
RESULTS_PATH = f"results/quantization/{DEVICE_NAME}/"
EXPERIMENT_NAME = "tensorflow"
REPS = 10
ITERS = 10
# ###################################################################


def compile_model(model, lr=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def xla_compile_model(model, lr=0.001):
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        jit_compile=True,
    )


def bench(model, reps, iters, input, xla: bool):
    if xla:
        print("Benchmarking xla verison...")
        compiled_tag = "-xla.csv"
    else:
        print("Benchmarking baseline...")
        compiled_tag = "-baseline.csv"

    fileName = RESULTS_PATH + EXPERIMENT_NAME + compiled_tag
    f = open(fileName, "w")
    results = []
    print_results = ""
    for i in range(reps):
        t = Timer(lambda: model.predict(input))
        print_results += str(t.timeit(number=iters)/iters) + ",\n"
        result = t.timeit(number=iters)/iters
        results.append(result)
        print(result)

    f.write(print_results)
    f.close()
    return results


def main():
    num_samples = 1
    input_shape = (224, 224, 3)
    random_input_data = np.random.rand(num_samples, *input_shape)
    # random_labels = np.random.randint(0, 1000, size=num_samples)

    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights="imagenet",
        classes=1000,
        classifier_activation="softmax",
    )
    xla_model = model

    compile_model(model)
    model.predict(random_input_data)  # warm up
    baseline_results = bench(model, REPS, ITERS, random_input_data, xla=False)

    tf.keras.backend.clear_session()

    xla_model.predict(random_input_data)  # warm up
    xla_compile_model(xla_model)
    xla_results = bench(xla_model, REPS, ITERS, random_input_data, xla=True)

    mean_baseline_latency = sum(baseline_results)/len(baseline_results)
    mean_xla_latency = sum(xla_results)/len(xla_results)

    print(f"The mean baseline e2e latency: {mean_baseline_latency}")
    print(f"The mean xla e2e latency: {mean_xla_latency}")

    model_path = "scripts/quantization/tensorflow_baseline_model.h5"
    tf.keras.models.save_model(model, model_path, include_optimizer=False)

    print("Size of gzipped baseline model: %.2f MiB" % (get_gzipped_model_size(model_path)))


if __name__ == "__main__":
    main()
