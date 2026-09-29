# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

'''
本代码为对megablocks(https://github.com/databricks/megablocks/blob/main/megablocks/backend/kernels.py)中padded_gather、padded_scatter以及padded_scatter_wgrad在Ascend上的亲和实现，以提升在Ascend上的性能
'''

import os
import sys
import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import is_npu, to_numpy, get_npu_properties, round_up


@triton.jit
def _padded_gather_kernel(
    x,
    out,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
        idx_mask = idx_offsets < idx_end

        cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
        cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask, other=0)
        base_bin_idx = extension.get_element(cur_bin_ids, (0,))  # expert_id
        tmp_buf = tl.zeros((SUB_BLOCK_SIZE, BLOCK_X), x.dtype.element_ty)
        for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
            col_offsets = tl.arange(0, BLOCK_X) + col_offset
            col_mask = col_offsets < NUM_COLUMNS
            # 计算当前token属于当前专家处理的第几个token
            offset_in_bin = idx_offset
            if base_bin_idx > 0:
                offset_in_bin -= tl.load(bins + base_bin_idx - 1)
            idx_o = offset_in_bin
            if base_bin_idx > 0:
                idx_o += tl.load(padded_bins + base_bin_idx - 1)
            COUNT = 0
            for i in range(0, SUB_BLOCK_SIZE):
                if idx_offset + i < idx_end:
                    bin_idx = extension.get_element(cur_bin_ids, (i,))
                    if bin_idx != base_bin_idx:   # 专家号变更时，需要把前一个专家处理的数据写回gm
                        # store前一个专家要处理的token
                        o_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_o
                        o_mask = o_offsets < idx_o + COUNT
                        tl.store(out + (o_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                 tmp_buf, o_mask[:, None] & col_mask[None, :])
                        # 更新起始偏移量和当前专家号
                        base_bin_idx = bin_idx
                        offset_in_bin = idx_offset + i
                        if base_bin_idx > 0:
                            offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                        idx_o = offset_in_bin
                        if base_bin_idx > 0:
                            idx_o += tl.load(padded_bins + base_bin_idx - 1)
                        COUNT = 0
                    idx_x = extension.get_element(cur_indices, (i,)) // TOP_K * NUM_COLUMNS
                    val = tl.load(x + idx_x + col_offsets, col_mask)
                    if SCALE:
                        scale = tl.load(weights + idx_x)
                        val = (val.to(tl.float32) * scale.to(tl.float32)).to(out.dtype.element_ty)
                    tmp_buf = extension.insert_slice(tmp_buf, val[None, :], offsets=(COUNT, 0), sizes=(1, BLOCK_X),
                                              strides=(1, 1))
                    COUNT += 1
            # store last block
            o_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_o
            o_mask = o_offsets < idx_o + COUNT
            tl.store(out + (o_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                     tmp_buf, o_mask[:, None] & col_mask[None, :])


def padded_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
):
    # create output buffer
    output_rows = padded_bins[-1].cpu().item()
    out = torch.zeros((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)

    # tiling
    # NOTE: tiling泛化，如果ub overflow，调小max_block_x和sub_block_size
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    max_block_x = 20000
    # 输入bf16/fp16, ub首地址需要32字节对齐
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    enable_multi_buffer = True
    sub_block_size = max((60 * 1024 - block_x * 2) // (block_x * 2 + 8), 1)

    scale = weights is not None
    _padded_gather_kernel[(num_core,)](
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        indices.shape[0],
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        TOP_K=top_k,
        SCALE=scale,
        multibuffer=enable_multi_buffer
    )
    return out


@triton.jit
def _padded_scatter_kernel(
    x,
    out,
    indices,
    bin_ids,
    weights,
    bins,
    padded_bins,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        # 跳过越界的空sub-block：此时cur_sub_block_size为负，
        # get_element负索引会读到垃圾bin id，导致标量访问GM越界(aivec error 264)
        if idx_offset < idx_end:
            idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
            idx_mask = idx_offsets < idx_end

            cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
            cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask, other=0)

            cur_sub_block_size = tl.minimum(SUB_BLOCK_SIZE, idx_end - idx_offset)
            # 统计当前sub_block_size中有多少个专家
            first_bin_idx = extension.get_element(cur_bin_ids, (0,))
            last_bin_idx = extension.get_element(cur_bin_ids, (cur_sub_block_size - 1,))
            bin_count = last_bin_idx - first_bin_idx + 1
            # 循环处理每个专家对应的数据
            begin = 0
            for i in range(bin_count):
                base_bin_idx = extension.get_element(cur_bin_ids, (begin,))  # 当前专家id
                # 找同一个专家的起止
                cur_bin = tl.load(bins + base_bin_idx)  # 当前专家及之前的专家一共要处理多少token
                count = tl.minimum(cur_bin - idx_offset - begin, cur_sub_block_size - begin)
                end = begin + count
                # 处理相同专家的数据
                offset_in_bin = idx_offset + begin
                if base_bin_idx > 0:
                    offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                x_offset = offset_in_bin
                if base_bin_idx > 0:
                    x_offset += tl.load(padded_bins + base_bin_idx - 1)
                x_offsets = tl.arange(0, SUB_BLOCK_SIZE) + x_offset
                x_mask = x_offsets < x_offset + (end - begin)

                for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                    col_offsets = tl.arange(0, BLOCK_X) + col_offset
                    col_mask = col_offsets < NUM_COLUMNS
                    cur_x = tl.load(x + (x_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                    x_mask[:, None] & col_mask[None, :])
                    for j in range(begin, end):
                        val_idx = j - begin
                        val = extension.extract_slice(cur_x, offsets=(val_idx, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                        idx = extension.get_element(cur_indices, (j,))
                        if SCALE:
                            scale = tl.load(weights + idx)
                            val = val.to(tl.float32) * scale.to(tl.float32)
                        tl.store(out + idx * NUM_COLUMNS + col_offsets,
                                 val.to(out.dtype.element_ty).reshape(BLOCK_X), col_mask)
                # 跳到下一个专家位置
                begin = end


def padded_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
):
    # create output buffer
    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)

    # tiling
    # NOTE: tiling泛化，如果ub overflow，调小max_block_x和sub_block_size
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    max_block_x = 6144
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    enable_multi_buffer = True
    sub_block_size = max((70 * 1024 - block_x * 12) // (block_x * 2 + 4), 1)
    
    scale = weights is not None
    _padded_scatter_kernel[(num_core,)](
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        indices.shape[0],
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        TOP_K=top_k,
        SCALE=scale,
        multibuffer=enable_multi_buffer
    )
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, x.shape[1])

@triton.jit
def _padded_scatter_wgrad_kernel(
    x,
    grads,
    out,
    indices,
    bin_ids,
    bins,
    padded_bins,
    INDICES_LENGTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
    NUM_COLUMNS: tl.constexpr,
    BLOCK_X: tl.constexpr,
    TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_begin = pid * BLOCK_SIZE
    idx_end = tl.minimum((pid + 1) * BLOCK_SIZE, INDICES_LENGTH)
    for idx_in_block in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        idx_offset = idx_begin + idx_in_block
        if SUB_BLOCK_SIZE > 1:
            # 跳过越界的空sub-block：此时cur_sub_block_size为负，
            # get_element负索引会读到垃圾bin id，导致标量访问GM越界(aivec error 264)
            if idx_offset < idx_end:
                idx_offsets = tl.arange(0, SUB_BLOCK_SIZE) + idx_offset
                idx_mask = idx_offsets < idx_end

                cur_indices = tl.load(indices + idx_offsets, idx_mask, other=0)
                cur_bin_ids = tl.load(bin_ids + idx_offsets, idx_mask, other=0)

                cur_sub_block_size = tl.minimum(SUB_BLOCK_SIZE, idx_end - idx_offset)
                # 统计当前sub_block_size中有多少个专家
                first_bin_idx = extension.get_element(cur_bin_ids, (0,))
                last_bin_idx = extension.get_element(cur_bin_ids, (cur_sub_block_size - 1,))
                bin_count = last_bin_idx - first_bin_idx + 1
                # 循环处理每个专家对应的数据
                begin = 0
                for i in range(bin_count):
                    base_bin_idx = extension.get_element(cur_bin_ids, (begin,))  # 当前专家id
                    # 找同一个专家的起止
                    cur_bin = tl.load(bins + base_bin_idx)  # 当前专家及之前的专家一共要处理多少token
                    count = tl.minimum(cur_bin - idx_offset - begin, cur_sub_block_size - begin)
                    end = begin + count
                    # 处理相同专家的数据
                    offset_in_bin = idx_offset + begin
                    if base_bin_idx > 0:
                        offset_in_bin -= tl.load(bins + base_bin_idx - 1)
                    x_offset = offset_in_bin
                    if base_bin_idx > 0:
                        x_offset += tl.load(padded_bins + base_bin_idx - 1)
                    x_offsets = tl.arange(0, SUB_BLOCK_SIZE) + x_offset
                    x_mask = x_offsets < x_offset + (end - begin)

                    col_offsets = tl.arange(0, BLOCK_X)
                    col_mask = col_offsets < NUM_COLUMNS
                    cur_x = tl.load(x + (x_offsets[:, None] * NUM_COLUMNS + col_offsets[None, :]),
                                    x_mask[:, None] & col_mask[None, :], other=0)
                    grad_offsets = tl.arange(0, BLOCK_X)
                    grad_mask = grad_offsets < NUM_COLUMNS
                    for j in range(begin, end):
                        val_idx = j - begin
                        val = extension.extract_slice(cur_x, offsets=(val_idx, 0), sizes=(1, BLOCK_X), strides=(1, 1))
                        idx = extension.get_element(cur_indices, (j,))
                        grad_ptr = grads + tl.multiple_of((idx // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
                        grad = tl.load(grad_ptr + grad_offsets, grad_mask, other=0)
                        acc = tl.sum(val.to(tl.float32).reshape(BLOCK_X) * grad.to(tl.float32))
                        tl.store(out + idx, acc.to(out.dtype.element_ty))
                    # 跳到下一个专家位置
                    begin = end
        elif idx_offset < idx_end:
            idx = tl.load(indices + idx_offset)
            bin_idx = tl.load(bin_ids + idx_offset)
            offset_in_bin = idx_offset
            if bin_idx > 0:
                offset_in_bin -= tl.load(bins + bin_idx - 1)
            x_offset = offset_in_bin
            if bin_idx > 0:
                x_offset += tl.load(padded_bins + bin_idx - 1)
            x_ptr = x + tl.multiple_of(x_offset * NUM_COLUMNS, NUM_COLUMNS)
            grad_ptr = grads + tl.multiple_of((idx // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
            acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
            for col_offset in range(0, NUM_COLUMNS, BLOCK_X):
                col_offsets = tl.arange(0, BLOCK_X) + col_offset
                col_masks = col_offsets < NUM_COLUMNS
                val = tl.load(x_ptr + col_offsets, col_masks, other=0)
                grad = tl.load(grad_ptr + col_offsets, col_masks, other=0)
                acc += val.to(tl.float32) * grad.to(tl.float32)
            acc = tl.sum(acc)
            tl.store(out + idx, acc)



def padded_scatter_wgrad(
    x: torch.Tensor,
    grads: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
):
    # create output buffer
    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    # tiling
    # NOTE: tiling泛化，如果ub overflow，调小max_block_x和sub_block_size
    num_core = get_npu_properties()["num_vectorcore"]
    indices_length = indices.shape[0]
    block_size = (indices_length - 1) // num_core + 1
    num_columns = x.shape[1]
    max_block_x = 5120
    block_x = (min(num_columns, max_block_x) + 15) // 16 * 16
    enable_multi_buffer = True
    sub_block_size = max((70 * 1024 - block_x * 12) // (block_x * 2 + 4), 1)
    if block_x < num_columns:
        sub_block_size = 1
    
    _padded_scatter_wgrad_kernel[(num_core,)](
        x,
        grads,
        out,
        indices,
        bin_ids,
        bins,
        padded_bins,
        indices.shape[0],
        block_size,
        sub_block_size,
        num_columns,
        block_x,
        TOP_K=top_k,
        multibuffer=enable_multi_buffer
    )
    return out


def test_fn(shape):
    assert is_npu()

    def generate_data(sl: int, hs: int, ne: int, top_k: int):
        # Create the data and indices.
        x = torch.randn((sl, hs)).npu().half()
        # Randomly assign tokens to experts.
        top_expert = torch.randint(0, ne, (sl * top_k,)).npu().int()
        bin_ids, indices = torch.sort(top_expert)
        tokens_per_expert = torch.histc(top_expert, ne, 0, ne - 1).to(torch.int32)
        padded_tokens_per_expert = round_up(tokens_per_expert, 128)
        padded_bins = torch.cumsum(padded_tokens_per_expert, dim=0).to(torch.int32)
        bins = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
        weights = torch.rand((sl * top_k,)).npu().half()
        grads =  torch.randn((sl, hs)).npu().half()
        return x, indices, bin_ids, bins, padded_bins, weights, grads

    def padded_gather_numpy(
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        x = to_numpy(x)
        indices = to_numpy(indices)
        bin_ids = to_numpy(bin_ids)
        bins = to_numpy(bins)
        padded_bins = to_numpy(padded_bins)

        out = np.zeros((padded_bins[-1], x.shape[-1]))
        in_idx = 0
        for i, end in enumerate(bins):
            out_idx = 0 if i == 0 else padded_bins[i - 1]
            while in_idx < end:
                load_idx = indices[in_idx] // top_k
                out[out_idx, :] = x[load_idx, :]
                in_idx += 1
                out_idx += 1
        return torch.from_numpy(out).npu().half()
    
    def padded_scatter_numpy(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        x = x.detach().cpu().numpy()
        indices: np.ndarray = to_numpy(indices)
        bin_ids: np.ndarray = to_numpy(bin_ids)
        weights: np.ndarray = to_numpy(weights)
        bins: np.ndarray = to_numpy(bins)
        padded_bins: np.ndarray = to_numpy(padded_bins)

        out = np.zeros((indices.shape[0] // top_k, x.shape[-1]))
        out_idx = 0
        for i, end in enumerate(bins):
            in_idx = 0 if i == 0 else padded_bins[i - 1]
            while out_idx < end:
                store_idx = indices[out_idx]
                scale = weights[store_idx]
                store_idx //= top_k

                out[store_idx, :] += scale * x[in_idx, :]
                out_idx += 1
                in_idx += 1
        return torch.from_numpy(out).npu().half()
    
    def padded_scatter_wgrad_numpy(
        x: torch.Tensor,
        grads: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        padded_bins: torch.Tensor,
        top_k: int,
    ):
        x = to_numpy(x)
        grads = to_numpy(grads)
        indices = to_numpy(indices)
        bin_ids = to_numpy(bin_ids)
        bins = to_numpy(bins)
        padded_bins = to_numpy(padded_bins)

        out = np.zeros(indices.shape).astype(np.float32)
        in_idx = 0
        for i,  end in enumerate(bins):
            x_idx = 0 if i == 0 else padded_bins[i - 1]
            while in_idx < end:
                data = x[x_idx, :]
                grad_idx = indices[in_idx] // top_k
                grad = grads[grad_idx]
                out_idx = indices[in_idx]
                out[out_idx] = np.sum(data.astype(np.float32) * grad.astype(np.float32))
                in_idx += 1
                x_idx += 1
        return torch.from_numpy(out).npu().half()


    top_k = shape[-1]
    x, indices, bin_ids, bins, padded_bins, weights, grads = generate_data(*shape)
    
    # padded_gather
    gather_ans = padded_gather_numpy(x, indices, bin_ids, bins, padded_bins, top_k)
    gather_results = padded_gather(x, indices, bin_ids, bins, padded_bins, None, top_k)
    assert torch.all(torch.eq(gather_ans, gather_results))

    # padded_scatter
    scatter_ans = padded_scatter_numpy(gather_ans, indices, weights,  bin_ids, bins, padded_bins, top_k)
    scatter_results = padded_scatter(gather_ans, indices, weights,  bin_ids, bins, padded_bins, top_k)
    assert torch.all(torch.eq(scatter_ans, scatter_results))

    # padded_scatter_wgrad
    scatter_wgrad_ans = padded_scatter_wgrad_numpy(gather_ans, grads, indices,  bin_ids, bins, padded_bins, top_k)
    scatter_wgrad_results = padded_scatter_wgrad(gather_ans, grads, indices,  bin_ids, bins, padded_bins, top_k)
    assert np.testing.assert_allclose(
                    to_numpy(scatter_wgrad_ans),
                    to_numpy(scatter_wgrad_results),
                    rtol=5e-2,
                ) is None


if __name__ == "__main__":
    shapes = (
        (1024, 4, 64, 4),
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 4),
        (16384, 768, 64, 4),
        (16384, 768, 128, 4)
    )

    for shape in shapes:
        test_fn(shape)
