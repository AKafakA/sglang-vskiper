"""Mapped decode-attention kernels extracted from the mirrored SGLang integration.

Kernel bodies and accumulation order are preserved from reference 461f0af45e.
"""
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_grouped_mapped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale_withk,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    active_row_map_ptr,
    active_row_count_ptr,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_buf_kpage,
    stride_buf_ktok,
    stride_buf_vpage,
    stride_buf_vtok,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    ACTIVE_ROW_WORKERS: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    HAS_MLA: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
    PAGE_SIZE: tl.constexpr = 1,
):
    row_worker = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    active_row_count = tl.load(active_row_count_ptr)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
    if not HAS_MLA:
        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]

    for row_slot in tl.range(
        row_worker, active_row_count, ACTIVE_ROW_WORKERS
    ):
        cur_batch = tl.load(active_row_map_ptr + row_slot)
        cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
        cur_batch_seq_len = (
            tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
        )
        kv_splits = tl.load(num_kv_splits + cur_batch)

        if xai_temperature_len > 0:
            offs_qidx = cur_batch_seq_len - 1
            xai_temperature_scale = 1.0 / tl.log2(
                float(xai_temperature_len)
            )
            _qtemp = (
                tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
            )
            xai_temperature_reg = tl.where(
                offs_qidx > xai_temperature_len, _qtemp, 1.0
            )

        offs_q = (
            cur_batch * stride_qbs
            + cur_head[:, None] * stride_qh
            + offs_d[None, :]
        )
        if BLOCK_DPE > 0:
            off_qpe = (
                cur_batch * stride_qbs
                + cur_head[:, None] * stride_qh
                + offs_dpe[None, :]
            )

        kv_len_per_split = (
            tl.cdiv(
                tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV
            )
            * MIN_BLOCK_KV
        )
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(
            split_kv_start + kv_len_per_split, cur_batch_seq_len
        )

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

        if split_kv_end > split_kv_start:
            q = tl.load(
                Q + offs_q,
                mask=(mask_h[:, None]) & (mask_d[None, :]),
                other=0.0,
            )
            q_k = q.to(K_Buffer.dtype.element_ty)
            if BLOCK_DPE > 0:
                qpe = tl.load(
                    Q + off_qpe,
                    mask=(mask_h[:, None]) & (mask_dpe[None, :]),
                    other=0.0,
                )
            for start_n in tl.range(
                split_kv_start, split_kv_end, BLOCK_N
            ):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                kv_loc = tl.load(
                    kv_indices + cur_batch_kv_start_idx + offs_n,
                    mask=offs_n < split_kv_end,
                    other=0,
                )
                if PAGE_SIZE == 1:
                    offs_buf_k = (
                        kv_loc[None, :] * stride_buf_kbs + base_offs_k
                    )
                else:
                    page_id = kv_loc // PAGE_SIZE
                    tok_in_p = kv_loc % PAGE_SIZE
                    offs_buf_k = (
                        page_id[None, :] * stride_buf_kpage
                        + tok_in_p[None, :] * stride_buf_ktok
                        + base_offs_k
                    )
                k = tl.load(
                    K_Buffer + offs_buf_k,
                    mask=(offs_n[None, :] < split_kv_end)
                    & (mask_d[:, None]),
                    other=0.0,
                )
                qk = tl.dot(q_k, k)
                if BLOCK_DPE > 0:
                    if PAGE_SIZE == 1:
                        offs_buf_kpe = (
                            kv_loc[None, :] * stride_buf_kbs
                            + base_offs_kpe
                        )
                    else:
                        offs_buf_kpe = (
                            page_id[None, :] * stride_buf_kpage
                            + tok_in_p[None, :] * stride_buf_ktok
                            + base_offs_kpe
                        )
                    kpe = tl.load(
                        K_Buffer + offs_buf_kpe,
                        mask=(offs_n[None, :] < split_kv_end)
                        & (mask_dpe[:, None]),
                        other=0.0,
                    )
                    qk += tl.dot(qpe, kpe.to(qpe.dtype))
                qk *= sm_scale_withk

                if logit_cap > 0:
                    qk = logit_cap * tanh(qk / logit_cap)
                if xai_temperature_len > 0:
                    qk *= xai_temperature_reg[:, None]

                qk = tl.where(
                    mask_h[:, None]
                    & (offs_n[None, :] < split_kv_end),
                    qk,
                    float("-inf"),
                )
                if HAS_MLA:
                    v = tl.trans(k)
                else:
                    if PAGE_SIZE == 1:
                        offs_buf_v = (
                            kv_loc[:, None] * stride_buf_vbs + base_offs_v
                        )
                    else:
                        offs_buf_v = (
                            page_id[:, None] * stride_buf_vpage
                            + tok_in_p[:, None] * stride_buf_vtok
                            + base_offs_v
                        )
                    v = tl.load(
                        V_Buffer + offs_buf_v,
                        mask=(offs_n[:, None] < split_kv_end)
                        & (mask_dv[None, :]),
                        other=0.0,
                    )

                n_e_max = tl.maximum(tl.max(qk, 1), e_max)
                re_scale = tl.exp(e_max - n_e_max)
                p = tl.exp(qk - n_e_max[:, None])
                acc *= re_scale[:, None]
                acc += tl.dot(p.to(v.dtype), v)
                e_sum = e_sum * re_scale + tl.sum(p, 1)
                e_max = n_e_max

            offs_mid_o = (
                cur_batch * stride_mid_ob
                + cur_head[:, None] * stride_mid_oh
                + split_kv_id * stride_mid_os
                + offs_dv[None, :]
            )
            tl.store(
                Att_Out + offs_mid_o,
                acc / e_sum[:, None],
                mask=(mask_h[:, None]) & (mask_dv[None, :]),
            )

            offs_mid_o_1 = (
                cur_batch * stride_mid_ob
                + cur_head * stride_mid_oh
                + split_kv_id * stride_mid_os
            ) // Lv
            tl.store(
                Att_Lse + offs_mid_o_1,
                e_max + tl.log(e_sum),
                mask=mask_h,
            )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _fwd_mapped_kernel_stage2(
    Mid_O,
    Mid_O_1,
    O,
    v_scale,
    kv_indptr,
    num_kv_splits,
    active_row_map_ptr,
    active_row_count_ptr,
    sink_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAX_KV_SPLITS: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    ACTIVE_ROW_WORKERS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    row_worker = tl.program_id(0)
    cur_head = tl.program_id(1)
    active_row_count = tl.load(active_row_count_ptr)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    for row_slot in tl.range(
        row_worker, active_row_count, ACTIVE_ROW_WORKERS
    ):
        cur_batch = tl.load(active_row_map_ptr + row_slot)
        cur_batch_seq_len = tl.load(
            kv_indptr + cur_batch + 1
        ) - tl.load(kv_indptr + cur_batch)
        kv_splits = tl.load(num_kv_splits + cur_batch)

        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

        offs_v = (
            cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
        )
        offs_logic = (
            cur_batch * stride_mid_ob + cur_head * stride_mid_oh
        ) // Lv
        kv_len_per_split = (
            tl.cdiv(
                tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV
            )
            * MIN_BLOCK_KV
        )

        for split_kv_id in tl.range(
            0, MAX_KV_SPLITS, num_stages=2
        ):
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end = tl.minimum(
                split_kv_start + kv_len_per_split, cur_batch_seq_len
            )

            if split_kv_end > split_kv_start:
                tv = tl.load(
                    Mid_O + offs_v + split_kv_id * stride_mid_os,
                    mask=mask_d,
                    other=0.0,
                )
                tlogic = tl.load(
                    Mid_O_1
                    + offs_logic
                    + split_kv_id * stride_mid_os // Lv
                )
                n_e_max = tl.maximum(tlogic, e_max)
                old_scale = tl.exp(e_max - n_e_max)
                acc *= old_scale
                exp_logic = tl.exp(tlogic - n_e_max)
                acc += exp_logic * tv
                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        if HAS_SINK:
            cur_sink = tl.load(sink_ptr + cur_head)
            e_sum += tl.exp(cur_sink - e_max)

        result = acc / e_sum * v_scale
        tl.store(
            O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
            result,
            mask=mask_d,
        )

