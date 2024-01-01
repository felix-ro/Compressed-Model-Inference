import tensorflow as tf
import timeit

inp = tf.random.normal((2, 64, 64, 3))


def conv2d():
    tf.keras.layers.Conv2D(16, 3, strides=1, padding='same')(inp)


def conv2dDepthwise():
    tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same')(inp)


def benchmark(input):
    global inp
    inp = input
    print("Input shape: " + str(inp.shape))

    # warum up
    conv2d()
    conv2dDepthwise()

    # measure
    num_iterations = 100
    conv2d_time = timeit.timeit(conv2d, number=num_iterations)
    conv2d_depthwise_time = timeit.timeit(conv2dDepthwise, number=num_iterations)

    conv2d_avg = conv2d_time / num_iterations
    conv2d_depthwise_avg = conv2d_depthwise_time / num_iterations

    print(f"Conv2D Time: {conv2d_avg} seconds per iteration")
    print(f"Conv2D Depthwise Time: {conv2d_depthwise_avg} seconds per iteration\n")

    return conv2d_avg, conv2d_depthwise_avg


def main():
    batch_sizes = []
    for i in range(65):
        batch_sizes.append(i)

    results_conv2d = []
    results_conv2d_depthwise = []

    for batch_size in batch_sizes:
        conv2d_avg, conv2d_depthwise_avg = benchmark(tf.random.normal((batch_size, 32, 32, 3)))
        results_conv2d.append(conv2d_avg * 1000)  # convert to ms
        results_conv2d_depthwise.append(conv2d_depthwise_avg * 1000)  # convert to ms

    with open("results/depthwise/conv2d.csv", "w") as f:
        f.write("Batch Size, Conv2d, Conv2d Depthwise\n")
        for i, batch_size in enumerate(batch_sizes):
            f.write(f"{batch_size}, {results_conv2d[i]}, {results_conv2d_depthwise[i]}\n")


if __name__ == "__main__":
    main()
