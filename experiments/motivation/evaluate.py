import tensorflow as tf

import train_distilled
from utils import getDataset
from speech_dataset import SpeechDataset


def main():
    dataset = getDataset()
    
    try: 
        model = tf.keras.models.load_model("model-uncompressedRes.h5")
    except Exception as e: 
        print("Could not find uncompressed model. Run train_uncompressed.py first.")
        return
    
    try: 
        student = train_distilled.getStudent()
        student.load_weights("student_model.h5")
        student.compile(metrics=["accuracy"])
    except Exception as e:
        print("Could not find distilled model. Run train_distilled.py first.")
        return
    

    test_data = dataset.testing_dataset().batch(1)
    student.evaluate(test_data)
    model.evaluate(test_data)


if __name__ == "__main__":
    main()