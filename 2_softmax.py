import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


def native_softmax(x: torch.Tensor):
    # input = M x N
    # R: MN         W: M        Flops:
    x_max = x.max(dim=1, keepdim=False)[0]  # shape: M
    # R: MN + M     W: MN       Flops: MN subtraction
    z = x - x_max[:, None]
    # R: MN         W: MN       Flops:
    numerator = torch.exp(z)
    # R: MN         W: M       Flops: MN ops
    denominator = numerator.sum(1)
    # R: MN + M     W: MN      Flops: MN ops
    out = numerator / denominator[:, None]
    return out


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=float('-inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        tl.store(output_row_start_ptr + col_offsets, softmax, mask=mask)


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties['multiprocessor_count']
NUM_REGS = properties['max_num_regs']
TOTAL_SMEM_PER_SM = properties['max_shared_mem']
WARP_SIZE = properties['warpSize']


def softmax_tri(x: torch.Tensor):
    assert x.ndim == 2
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 << ((BLOCK_SIZE >= 2048) + (BLOCK_SIZE >= 4096))  # 4,8,16
    num_stages = 2 << (TOTAL_SMEM_PER_SM >= 200_000)
    y = torch.empty_like(x)
    kernel = _softmax_kernel.warmup(
        x,
        y,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SMEM_PER_SM // (sram_needed_per_program)
    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(programs_per_sm * NUM_SM, n_rows)
    grid = (num_programs, 1, 1)
    # kernel[grid](x, y, x.stride(0), y.stride(0), n_rows, n_cols)
    _softmax_kernel[grid](x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name="softmax-performance",
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_tri(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


def test_native_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device=DEVICE):
    assert type(size) is tuple and len(size) == 2
    #
    torch.manual_seed(0)
    x = torch.randn(size, device=device)
    z_manual = softmax_tri(x)
    z_ref = torch.softmax(x, dim=1)
    torch.testing.assert_close(z_ref, z_manual, atol=atol, rtol=rtol)
    print("Softmax test passed!")


if __name__ == "__main__":
    test_native_softmax_kernel((1024, 512))
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path="./benchmarks", print_data=False)
