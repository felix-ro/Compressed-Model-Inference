import torch


def prof(f, use_cuda=True):
    with torch.profiler.profile(record_shapes=True, use_cuda=use_cuda, profile_memory=True, with_flops=True, with_stack=True, with_modules=True) as prof:
        with torch.profiler.record_function("f"):
            for i in range(10):
                f()
        torch.cuda.synchronize()
    return prof.key_averages().table()

def main():
    torch.backends.cudnn.benchmark=True
    t = torch.randn(16, 128, 256, 256, device='cuda')
    results = []
    for i in range(8):
        conv = torch.nn.Conv2d(128, 128, 3, groups=2**i).cuda()
        results.append([2**i, prof(lambda : conv(t))])
    for i, r in results:
        print(f"groups={i}", r.splitlines()[-2:])

if __name__ == "__main__":
    main()
