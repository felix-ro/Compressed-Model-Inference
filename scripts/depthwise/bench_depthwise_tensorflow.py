import tensorflow as tf
import timeit

input = tf.random.normal((2, 64, 64, 3))
kernel_size = 1


def conv2d():
    tf.keras.layers.Conv2D(filters=16, kernel_size=kernel_size, strides=1, padding='same')(input)


def conv2dSeparable():
    tf.keras.layers.SeparableConv2D(filters=16, kernel_size=kernel_size, strides=1, padding='same')(input)


def benchmark(inp, kern_size):
    global input, kernel_size
    input = inp
    kernel_size = kern_size
    print(f"Input shape: {input.shape}")
    print(f"Kernel size: {kernel_size}")

    # measure
    num_iterations = 100

    conv2d()  # warm up
    conv2d_time = timeit.timeit(conv2d, number=num_iterations)

    conv2dSeparable()  # warm up
    conv2d_separable_time = timeit.timeit(conv2dSeparable, number=num_iterations)

    conv2d_avg = conv2d_time / num_iterations
    conv2d_separable_avg = conv2d_separable_time / num_iterations

    print(f"Conv2D Time: {conv2d_avg} seconds per iteration")
    print(f"Conv2D Separable Time: {conv2d_separable_avg} seconds per iteration\n")

    return conv2d_avg, conv2d_separable_avg


def main():
    kernel_sizes = [1, 2, 3]
    batch_sizes = []
    for i in range(65):
        batch_sizes.append(i)

    results_conv2d_kernels = []
    results_conv2d_separable_kernels = []
    for kern_size in kernel_sizes:
        results_conv2d = []
        results_conv2d_separable = []
        for batch_size in batch_sizes:
            conv2d_avg, conv2d_separable_avg = \
                  benchmark(tf.random.normal((batch_size, 32, 32, 3)), kern_size)

            results_conv2d.append(conv2d_avg * 1000)  # convert to ms
            results_conv2d_separable.append(conv2d_separable_avg * 1000)  # convert to ms

        results_conv2d_kernels.append(results_conv2d)
        results_conv2d_separable_kernels.append(results_conv2d_separable)

    with open("results/depthwise/conv2d.csv", "w") as f:
        column_names = "Batch Size"
        for i in kernel_sizes:
            column_names += f",Conv2D {i}x{i}"
            column_names += f",Separable Conv2D {i}x{i}"
        f.write(column_names + "\n")

        for j, batch_size in enumerate(batch_sizes):
            line = f"{batch_size}"
            for i, _ in enumerate(kernel_sizes):
                line += f",{results_conv2d_kernels[i][j]},{results_conv2d_separable_kernels[i][j]}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    main()
