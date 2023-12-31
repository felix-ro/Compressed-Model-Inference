import tensorflow as tf
import timeit


conv2d = tf.keras.layers.Conv2D(16, 3, strides=1, padding='same')
conv2d_depthwise = tf.keras.layers.DepthwiseConv2D(3, padding='same')
inp = tf.random.normal((2, 64, 64, 3))


def timeConv2d():
    conv2d(inp)


def timeConv2dDepthwise():
    conv2d_depthwise(inp)


def benchmark(input):
    global inp
    inp = input
    print("Input shape: " + str(inp.shape))

    # warum up
    conv2d(inp)
    conv2d_depthwise(inp)

    # measure
    num_iterations = 100
    conv2d_time = timeit.timeit(timeConv2d, number=num_iterations)
    conv2d_depthwise_time = timeit.timeit(timeConv2dDepthwise, number=num_iterations)

    conv2d_avg = conv2d_time / num_iterations
    conv2d_depthwise_avg = conv2d_depthwise_time / num_iterations

    print(f"Conv2D Time: {conv2d_avg} seconds per iteration")
    print(f"Conv2D Depthwise Time: {conv2d_depthwise_avg} seconds per iteration\n")

    return conv2d_avg, conv2d_depthwise_avg


def main():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    results_conv2d = []
    results_conv2d_depthwise = []

    for batch_size in batch_sizes:
        conv2d_avg, conv2d_depthwise_avg = benchmark(tf.random.normal((batch_size, 64, 64, 3)))
        results_conv2d.append(conv2d_avg * 1000)  # convert to ms
        results_conv2d_depthwise.append(conv2d_depthwise_avg * 1000)  # convert to ms

    with open("results/depthwise/conv2d.csv", "w") as f:
        f.write("Batch Size, Conv2d [ms]\n")
        for i, batch_size in enumerate(batch_sizes):
            f.write(f"{batch_size}, {results_conv2d[i]}\n")

    with open("results/depthwise/conv2d_depthwise.csv", "w") as f:
        f.write("Batch Size, Conv2d Depthwise [ms]\n")
        for i, batch_size in enumerate(batch_sizes):
            f.write(f"{batch_size}, {results_conv2d_depthwise[i]}\n")


if __name__ == "__main__":
    main()
