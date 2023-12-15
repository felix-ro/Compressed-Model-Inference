import tensorflow as tf

import train_distilled
from speech_dataset import SpeechDataset


def main():
    dataset = SpeechDataset(words=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                            upper_band_limit=5000.0,  # ~ human voice range
                            lower_band_limit=125.0,
                            feature_bin_count=25,
                            window_size_ms=40.0,
                            window_stride=20.0,
                            silence_percentage=3, unknown_percentage=3)
    
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
    model.evaluate(test_data)
    student.evaluate(test_data)


if __name__ == "__main__":
    main()