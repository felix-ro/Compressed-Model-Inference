import torch as t
import timeit

conv2d = t.nn.Conv2d(32,32,3,1,1).cuda()
conv2d_depthwise = t.nn.Conv2d(32,32,3,1,1,groups=32).cuda()
inp = t.randn(2,32,512,512).cuda()

def timeConv2d():
    t.cuda.synchronize()
    conv2d(inp)
    t.cuda.synchronize()

def timeConv2dDepthwise():
    t.cuda.synchronize()
    conv2d_depthwise(inp)
    t.cuda.synchronize()

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

    t.backends.cudnn.benchmark=True

    benchmark(t.randn(2,32,512,512).cuda())
    # benchmark(t.randn(2,32,512,512).cuda())
    # benchmark(t.randn(2,32,512,512).cuda())


if __name__ == "__main__":
    main()