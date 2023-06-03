#ifndef _TOOLS_H_
#define _TOOLS_H_

#include <hacl/hacl.h>
#include "hacl_type.h"

constexpr int64_t UB_MAX_HALF_SIZE = 256 * 1024 / 2;
constexpr int64_t L0AB_MAX_HALF_SIZE = 64 * 1024 / 2;
constexpr int64_t L0C_MAX_SINGLE_SIZE = 256 * 1024 / 4;
constexpr int64_t L0C_MAX_HALF_SIZE = 256 * 1024 / 2;

constexpr int64_t L1_MAX_HALF_SIZE = 1024 * 1024 / 2;

constexpr int64_t UB_HALF_64KB = 64 * 1024 / 2;

HACL_INLINE __aicore__ void __memcpy(__gm__ half *gm, __ub__ half *ub, int64_t len)
{
    if (len < 16)
        len = 16;
    int64_t nUnit = len / 16;
    int64_t unit_remain = len % 16;
    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(gm, ub, 1, nUnit, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, 0);
    wait_flag(PIPE_MTE3, PIPE_S, 0);
    if (unit_remain && len > 16)
    {
        int64_t offset_start = (len & 0xFFFFFFFFFFFFFFF0) - 16 + unit_remain;
        for (int i = 0; i < 16; ++i)
        {
            *(ub + i) = *(ub + offset_start + i);
        }
        // set_flag(PIPE_S, PIPE_MTE3, 0);
        // wait_flag(PIPE_S, PIPE_MTE3, 0);
        __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(gm + offset_start, ub, 1, 1, 0, 0);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrixC_ND2zN(__ub__ half *ub_buffer1,
                                                      __ub__ half *ub_buffer0,
                                                      int64_t m_real_pad,
                                                      int64_t n_real_pad,
                                                      half beta)
{
    __ub__ half *dst_list[16], *src_list[16];
    for (int i = 0; i < 16; ++i)
    {
        dst_list[i] = ub_buffer1 + i * 16;
        src_list[i] = ub_buffer0 + i * m_real_pad;
    }
    for (int j = 0; j < n_real_pad / 16; ++j)
    {
        vec_trans_scatter(dst_list, src_list, m_real_pad / 16, 16, 1);
        for (int jj = 0; jj < 16; ++jj)
        {
            dst_list[jj] += m_real_pad * 16;
            src_list[jj] += m_real_pad * 16;
        }
    }
    vec_muls(ub_buffer1, ub_buffer1, beta, 2 * UB_HALF_64KB);
}

HACL_INLINE __aicore__ void hablas_load_matrixC_ub2l0(__l0c__ float *l0c,
                                                      __ub__ half *ub_buffer1,
                                                      int64_t m_real_pad,
                                                      int64_t n_real_pad)
{
    _memcpy(l0c, ub_buffer1, 1, (m_real_pad / 16) * (n_real_pad / 16), 0, 0, block_t::MATRIX);
}

HACL_INLINE __aicore__ void hablas_store_matrixC_l02ub2ub(__ub__ half *ub_buffer1,
                                                          __ub__ half *ub_buffer0,
                                                          __l0c__ float *l0c,
                                                          int64_t m_real_pad,
                                                          int64_t n_real_pad)
{
    _memcpy(ub_buffer0, l0c, 1, (m_real_pad / 16) * (n_real_pad / 16), 0, 0);
    __ub__ half *dst_list[16], *src_list[16];
    for (int i = 0; i < 16; ++i)
    {
        dst_list[i] = ub_buffer1 + i * m_real_pad;
        src_list[i] = ub_buffer0 + i * 16;
    }
    for (int j = 0; j < n_real_pad / 16; ++j)
    {
        vec_trans_scatter(dst_list, src_list, m_real_pad / 16, 1, 16);
        for (int jj = 0; jj < 16; ++jj)
        {
            dst_list[jj] += m_real_pad * 16;
            src_list[jj] += m_real_pad * 16;
        }
    }
}

HACL_INLINE __aicore__ void hablas_store_matrixC_l02ub(__ub__ half *ub_buffer0,
                                                       __l0c__ float *l0c,
                                                       int64_t m_real_pad,
                                                       int64_t n_real_pad)
{
    _memcpy(ub_buffer0, l0c, 1, (m_real_pad / 16) * (n_real_pad / 16), 0, 0);
}

HACL_INLINE __aicore__ void hablas_store_matrixC_nZ2ND(__ub__ half *ub_buffer1,
                                                       __ub__ half *ub_buffer0,
                                                       int64_t m_real_pad,
                                                       int64_t n_real_pad)
{
    __ub__ half *dst_list[16], *src_list[16];
    for (int i = 0; i < 16; ++i)
    {
        dst_list[i] = ub_buffer1 + i * m_real_pad;
        src_list[i] = ub_buffer0 + i * 16;
    }
    for (int j = 0; j < n_real_pad / 16; ++j)
    {
        vec_trans_scatter(dst_list, src_list, m_real_pad / 16, 1, 16);
        for (int jj = 0; jj < 16; ++jj)
        {
            dst_list[jj] += m_real_pad * 16;
            src_list[jj] += m_real_pad * 16;
        }
    }
}

HACL_INLINE __aicore__ void hablas_store_matrixC_ub2gm(__gm__ half *gm,
                                                       __ub__ half *ub_buffer1,
                                                       int64_t m_real_pad,
                                                       int64_t n_real_pad,
                                                       int64_t m_real,
                                                       int64_t n_real,
                                                       int64_t ldc)
{
    if (m_real % 16 || (ldc - m_real) % 16)
    {
        if (m_real != m_real_pad)
        {
            for (int j = 0; j < n_real; ++j)
            {
                __memcpy(gm + j * ldc, ub_buffer1 + j * m_real_pad, m_real);
            }
        }
        else
        {
            int64_t remain = n_real % 16;
            int64_t nburst = n_real / 16;
            if (nburst)
                for (int j = 0; j < 16; ++j)
                {
                    _memcpy(gm + j * ldc, ub_buffer1 + j * m_real_pad, nburst, m_real / 16, (ldc * 16 - m_real) / 16, (m_real * 16 - m_real) / 16);
                }
            for (int j = 0; j < remain; ++j)
            {
                __memcpy(gm + (nburst * 16 + j) * ldc, ub_buffer1 + (nburst * 16 + j) * m_real_pad, m_real);
            }
        }
    }
    else
    {
        _memcpy(gm, ub_buffer1, n_real, m_real / 16, (ldc - m_real) / 16, 0);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_gm2ub(__ub__ half *ub_buffer0,
                                                     __gm__ half *gm,
                                                     int64_t m_real_pad,
                                                     int64_t n_real_pad,
                                                     int64_t m_real,
                                                     int64_t n_real,
                                                     int64_t stride)
{
    if (m_real % 16 || (stride - m_real) % 16)
    {
        if (m_real != m_real_pad)
        {
            for (int j = 0; j < n_real; ++j)
            {
                _memcpy(ub_buffer0 + j * m_real_pad, gm + j * stride, 1, m_real_pad / 16, 0, 0);
            }
        }
        else
        {
            int64_t remain = n_real % 16;
            int64_t nburst = n_real / 16;
            if (nburst)
                for (int j = 0; j < 16; ++j)
                {
                    _memcpy(ub_buffer0 + j * m_real_pad, gm + j * stride, nburst, m_real / 16, (m_real * 16 - m_real) / 16, (stride * 16 - m_real) / 16);
                }
            for (int j = 0; j < remain; ++j)
            {
                _memcpy(ub_buffer0 + (nburst * 16 + j) * m_real_pad, gm + (nburst * 16 + j) * stride, 1, m_real_pad / 16, 0, 0);
            }
        }
    }
    else
    {
        _memcpy(ub_buffer0, gm, n_real, m_real / 16, 0, (stride - m_real) / 16);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_gm2l1(__l1__ half *l1_buffer0,
                                                     __gm__ half *gm,
                                                     int64_t m_real_pad,
                                                     int64_t n_real_pad,
                                                     int64_t m_real,
                                                     int64_t n_real,
                                                     int64_t stride)
{
    if (m_real % 16 || (stride - m_real) % 16)
    {
        if (m_real != m_real_pad)
        {
            for (int j = 0; j < n_real; ++j)
            {
                _memcpy(l1_buffer0 + j * m_real_pad, gm + j * stride, 1, m_real_pad / 16, 0, 0);
            }
        }
        else
        {
            int64_t remain = n_real % 16;
            int64_t nburst = n_real / 16;
            if (nburst)
                for (int j = 0; j < 16; ++j)
                {
                    _memcpy(l1_buffer0 + j * m_real_pad, gm + j * stride, nburst, m_real / 16, (m_real * 16 - m_real) / 16, (stride * 16 - m_real) / 16);
                }
            for (int j = 0; j < remain; ++j)
            {
                _memcpy(l1_buffer0 + (nburst * 16 + j) * m_real_pad, gm + (nburst * 16 + j) * stride, 1, m_real_pad / 16, 0, 0);
            }
        }
    }
    else
    {
        _memcpy(l1_buffer0, gm, n_real, m_real / 16, 0, (stride - m_real) / 16);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_l12ub(__ub__ half *ub,
                                                     __l1__ half *l1,
                                                     int64_t m_real_pad,
                                                     int64_t n_real_pad)
{
    _memcpy(ub, l1, 1, m_real_pad * n_real_pad / 16, 0, 0);
}

HACL_INLINE __aicore__ void hablas_load_input_matrix_ND2zZ(__ub__ half *ub_buffer1,
                                                           __ub__ half *ub_buffer0,
                                                           int64_t m_real_pad,
                                                           int64_t n_real_pad,
                                                           half alpha)
{
    __hacl_details__::__hacl_intrinsic_move_mask(128);
    for (int j = 0; j < m_real_pad / 16; ++j)
    {
        __hacl_details__::__hacl_intrinsic_vec_muls(ub_buffer1 + n_real_pad * 16 * j,
                                                    ub_buffer0 + 16 * j,
                                                    alpha,
                                                    n_real_pad / 16 * 2,
                                                    m_real_pad / 2, 8,
                                                    m_real_pad / 16, 1);
    }
}

HACL_INLINE __aicore__ void hablas_load_input_matrix_ub2l1(__l1__ half *l1,
                                                           __ub__ half *ub,
                                                           int64_t m_real_pad,
                                                           int64_t n_real_pad)
{
    _memcpy(l1, ub, 1, m_real_pad * n_real_pad / 16, 0, 0);
}

HACL_INLINE __aicore__ void hablas_load_l12l0a(__l0a__ half *l0a,
                                               __l1__ half *l1,
                                               int64_t m_real_pad,
                                               int64_t n_real_pad,
                                               hablasOperation_t trans)
{
    if (trans == HABLAS_OP_T)
    {
        for (int i = 0; i < n_real_pad / 16; ++i)
        {
            load2d(l0a + i * m_real_pad * 16, l1, i, m_real_pad / 16, n_real_pad / 16, false);
        }
    }
    else
    {
        load2d(l0a, l1, 0, (m_real_pad / 16) * (n_real_pad / 16), 1, true);
    }
}

HACL_INLINE __aicore__ void hablas_load_l12l0b(__l0b__ half *l0b,
                                               __l1__ half *l1,
                                               int64_t m_real_pad,
                                               int64_t n_real_pad,
                                               hablasOperation_t trans)
{
    if (trans == HABLAS_OP_T)
    {
        for (int i = 0; i < n_real_pad / 16; ++i)
        {
            load2d(l0b + i * m_real_pad * 16, l1, i, m_real_pad / 16, n_real_pad / 16, true);
        }
    }
    else
    {
        load2d(l0b, l1, 0, (m_real_pad / 16) * (n_real_pad / 16), 1, false);
    }
}

#endif
