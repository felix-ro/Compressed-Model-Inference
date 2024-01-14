import tensorflow as tf
from tensorflow import keras

from utils import getDataset
import train_uncompressed


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


def getStudent():
    student = keras.Sequential(
        [
            keras.Input(shape=(49, 25, 1)),
            keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

            keras.layers.Flatten(),
            keras.layers.Dense(12, activation='softmax'),
        ],
        name="student",
    )

    return student


def main():
    dataset = getDataset()
    batch_size = 128
    train_data = dataset.training_dataset().batch(batch_size).prefetch(1)

    try:
        student = getStudent()
        student.load_weights("student-model.h5")
    except Exception:
        print("No pretrained weights for student found. Distilling model...")
        try:
            model = tf.keras.models.load_model("model-uncompressedRes.h5")
        except Exception:
            print("Uncompressed model not found. Training uncompressed model first!")
            train_uncompressed.main()

        # Initialize and compile distiller
        student = getStudent()
        distiller = Distiller(student=student, teacher=model)
        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.1,
            temperature=10,
        )

        # Distill teacher to student
        student.summary()
        distiller.fit(train_data, epochs=35)
        student = distiller.student

    student.compile(metrics=["accuracy"])
    student.save_weights("student-model.h5")

    test_data = dataset.testing_dataset().batch(64)
    student.evaluate(test_data)


if __name__ == "__main__":
    main()
