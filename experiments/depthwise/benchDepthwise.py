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
    o = conv2d(inp)
    o = conv2d_depthwise(inp)

    # measure
    num_iterations = 100  
    conv2d_time = timeit.timeit(timeConv2d, number=num_iterations)
    conv2d_depthwise_time = timeit.timeit(timeConv2dDepthwise, number=num_iterations)

    # print results
    print(f"Conv2D Time: {conv2d_time / num_iterations} seconds per iteration")
    print(f"Conv2D Depthwise Time: {conv2d_depthwise_time / num_iterations} seconds per iteration\n")

def main():
    benchmark(tf.random.normal((2, 64, 64, 3)))
    benchmark(tf.random.normal((100, 64, 64, 3)))
    benchmark(tf.random.normal((1000, 8, 8, 3)))


if __name__ == "__main__":
    main()
    