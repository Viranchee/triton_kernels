"""
basics
test
benchmark
"""

import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def _add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    PID = tl.program_id(axis=0)
    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # load data from HBM to SRAM
    x = tl.load(x_ptr + offsets, mask=mask, other=None)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add_tri(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    # check tensors are on same device
    assert x.device == y.device == DEVICE
    # define our launch grid
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)  # for 4096 inputs, 4,0

    _add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def test_add_kernel(size: int, atol=1e-3, rtol=1e-3, device: torch.device = DEVICE):
    torch.manual_seed(0)
    x = torch.randn(size, device=device)
    y = torch.randn(size, device=device)
    # run Pytorch and Triton add kernel
    z_triton = add_tri(x, y)
    z_torch = x + y
    torch.testing.assert_close(z_triton, z_torch, atol=atol, rtol=rtol)
    print("Triton add kernel passed")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 30, 1)],
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark_add_kernel(size: int, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_mis, max_mis = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_mis, max_mis = triton.testing.do_bench(lambda: add_tri(x, y), quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_mis), gbps(min_mis)


if __name__ == "__main__":
    test_add_kernel(1024 * 4)
    test_add_kernel(1024 * 4 + 1)
    test_add_kernel(98432)

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark_add_kernel.run(save_path='./benchmarks', print_data=True)
