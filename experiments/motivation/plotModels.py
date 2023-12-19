import tensorflow as tf
from train_distilled import getStudent

def main():
    modelNames = ["model-uncompressedRes", "model-depthwise", "student-model"]

    for modelName in modelNames:
        if modelName == "student-model":
            student = getStudent()
            tf.keras.utils.plot_model(student, show_shapes=True, show_layer_names=True, to_file=modelName + ".png")
        else: 
            try: 
                modelUncompressed = tf.keras.models.load_model(modelName + ".h5")
            except Exception as e:
                print("Model \'model-uncompressedRes.h5\' was not found!")

            tf.keras.utils.plot_model(modelUncompressed, show_shapes=True, show_layer_names=True, to_file=modelName + ".png")


if __name__ == "__main__":
    main()