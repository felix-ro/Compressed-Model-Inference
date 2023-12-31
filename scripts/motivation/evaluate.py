import tensorflow as tf

import train_distilled
from utils import getDataset


def main():
    dataset = getDataset()

    try:
        model = tf.keras.models.load_model("model-uncompressedRes.h5")
    except Exception:
        print("Could not find uncompressed model. Run train_uncompressed.py first.")
        return

    try:
        student = train_distilled.getStudent()
        student.load_weights("student-model.h5")
        student.compile(metrics=["accuracy"])
    except Exception:
        print("Could not find distilled model. Run train_distilled.py first.")
        return

    try:
        modelDepthwise = tf.keras.models.load_model("model-depthwise.h5")
    except Exception:
        print("Could not find compressed depthwise model. Run train_depthwise.py first.")
        return

    test_data = dataset.testing_dataset().batch(4482).prefetch(1)
    print("Base Model:")
    model.evaluate(test_data)
    print("Distilled Model:")
    student.evaluate(test_data)
    print("Depthwise Model:")
    modelDepthwise.evaluate(test_data)


if __name__ == "__main__":
    main()
