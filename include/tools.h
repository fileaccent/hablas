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
                                                       __ub__ half *workspace,
                                                       int64_t m_real_pad,
                                                       int64_t n_real_pad,
                                                       int64_t m_real,
                                                       int64_t n_real,
                                                       int64_t ldc)
{
    if (m_real < 16)
    {
        int offset = 16 - m_real;
        _memcpy(workspace, gm, 1, 1, 0, 0);
        pipe_barrier(PIPE_ALL);
        for (int j = 0; j < m_real; ++j)
        {
            *(workspace + j) = *(ub_buffer1 + j);
        }
        set_flag(PIPE_S, PIPE_MTE3, 0);
        wait_flag(PIPE_S, PIPE_MTE3, 0);
        _memcpy(gm, workspace, 1, 1, 0, 0);
        set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        for (int i = 1; i < n_real; ++i)
        {
            wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
            _memcpy(workspace, gm + i * ldc - offset, 1, 1, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, 0);
            wait_flag(PIPE_MTE2, PIPE_S, 0);
            for (int j = 0; j < m_real; ++j)
            {
                *(workspace + offset + j) = *(ub_buffer1 + i * m_real_pad + j);
            }
            set_flag(PIPE_S, PIPE_MTE3, 0);
            wait_flag(PIPE_S, PIPE_MTE3, 0);
            _memcpy(gm + i * ldc - offset, workspace, 1, 1, 0, 0);
            set_flag(PIPE_MTE3, PIPE_MTE2, 0);
            pipe_barrier(PIPE_MTE3);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
    }
    else if (m_real % 16 || (ldc - m_real) % 16)
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

HACL_INLINE __aicore__ void hablas_load_Vector_gm2ub(__ub__ half *dst,
                                                     __gm__ half *src,
                                                     __ub__ half *temp,
                                                     int32_t len,
                                                     int32_t stride)
{
    int content = 32 * 1024;
    if (stride == 1)
    {
        _memcpy(dst, src, len);
    }
    else
    {
        int32_t loop = (stride * len) / content;
        int start_posi = 0;
        int iub = 0;
        for (int i = 0; i < loop; i++)
        {
            _memcpy(temp, src + i * content, content);
            set_flag(PIPE_MTE2, PIPE_S, 2);
            wait_flag(PIPE_MTE2, PIPE_S, 2);
            int iwhile = start_posi;
            while (iwhile < content)
            {
                *(dst + iub) = *(temp + iwhile);
                iwhile = iwhile + stride;
                iub = iub + 1;
            }
            start_posi = iwhile - content;
            set_flag(PIPE_S, PIPE_MTE2, 2);
            wait_flag(PIPE_S, PIPE_MTE2, 2);
        }
        if ((stride * len) % content)
        {
            _memcpy(temp, src + loop * content, (stride * len) % content);
            set_flag(PIPE_MTE2, PIPE_S, 2);
            wait_flag(PIPE_MTE2, PIPE_S, 2);
            int iwhile = start_posi;
            while (iub < len)
            {
                *(dst + iub) = *(temp + iwhile);
                iwhile = iwhile + stride;
                iub = iub + 1;
            }
        }
    }
    set_flag(PIPE_MTE2, PIPE_S, 2);
    wait_flag(PIPE_MTE2, PIPE_S, 2);
    if (len % 16)
    {
        int32_t r = len % 16;
        for (int i = 0; i < 16 - r; ++i)
        {
            *(dst + len + i) = (half)0.0;
        }
    }
}

HACL_INLINE __aicore__ void hablas_store_Vector_ub2gm(__gm__ half *dst,
                                                      __ub__ half *src,
                                                      __ub__ half *temp,
                                                      int32_t len,
                                                      int32_t valid_len,
                                                      int32_t stride)
{
    int content = 32 * 1024;
    if (stride == 1)
    {
        _memcpy(dst, src, len);
    }
    else
    {
        int32_t loop = (stride * valid_len) / content;
        int start_posi = 0;
        int iub = 0;
        set_flag(PIPE_V, PIPE_MTE2, 3);
        wait_flag(PIPE_V, PIPE_MTE2, 3);
        for (int i = 0; i < loop; i++)
        {
            _memcpy(temp, dst + i * content, content);
            set_flag(PIPE_MTE2, PIPE_S, 3);
            wait_flag(PIPE_MTE2, PIPE_S, 3);
            int iwhile = start_posi;
            while (iwhile < content)
            {
                *(temp + iwhile) = *(src + iub);
                iwhile = iwhile + stride;
                iub = iub + 1;
            }
            start_posi = iwhile - content;

            set_flag(PIPE_S, PIPE_MTE3, 3);
            wait_flag(PIPE_S, PIPE_MTE3, 3);

            _memcpy(dst + i * content, temp, content);
            set_flag(PIPE_MTE3, PIPE_MTE2, 3);
            wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
        }
        if ((stride * valid_len) % content)
        {
            _memcpy(temp, dst + loop * content, (stride * len) % content);
            set_flag(PIPE_MTE2, PIPE_S, 3);
            wait_flag(PIPE_MTE2, PIPE_S, 3);
            int iwhile = start_posi;
            while (iub < valid_len)
            {
                *(temp + iwhile) = *(src + iub);
                iwhile = iwhile + stride;
                iub = iub + 1;
            }
            set_flag(PIPE_S, PIPE_MTE3, 3);
            wait_flag(PIPE_S, PIPE_MTE3, 3);
            _memcpy(dst + loop * content, temp, (stride * len) % content);
        }
    }
}

HACL_INLINE __aicore__ void hablas_load_Matrix_dig_upper(__ub__ half *dst,
                                                         __ub__ half *mask,
                                                         __ub__ half *mask_dia,
                                                         int n_real_pad)
{
    Vector<half_16, 16, HACL_UB> ub_temp;
    Vector<half_16, 16, HACL_UB> ub_tran;

    int n_loop = n_real_pad / 16 ;
     
    for(int i = 0; i < n_loop; i++){
    	for(int j = 0; j < i; j++){
    		vec_trans(dst + i * n_real_pad * 16 + j * 256, dst + j * n_real_pad * 16 + i * 256, 1, 1, 1);
    	}
    }
     
    // pipe_barrier(PIPE_V);
    for(int a = 0; a < n_loop; a++)
    {
        vec_mul(ub_temp.get_ptr(0), dst + a * n_real_pad * 16 + a * 256, mask, 256);
        vec_mul(ub_tran.get_ptr(0), ub_temp.get_ptr(0), mask_dia, 256);
        pipe_barrier(PIPE_V);
        vec_trans(ub_tran.get_ptr(0), ub_tran.get_ptr(0), 1, 1, 1);
        vec_add(dst + a * n_real_pad * 16 + a * 256, ub_temp.get_ptr(0), ub_tran.get_ptr(0), 256);
    }
}


HACL_INLINE __aicore__ void hablas_load_Matrix_dig_lower(__ub__ half *dst,
                                                         __ub__ half *mask,
                                                         __ub__ half *mask_dia,
                                                         int n_real_pad)
{
    Vector<half_16, 16, HACL_UB> ub_temp;
    Vector<half_16, 16, HACL_UB> ub_tran;

    int n_loop = n_real_pad / 16 ;
    
    for(int i = 0; i < n_loop; i++){
    	for(int j = i + 1; j < n_loop; j++){
    		vec_trans(dst + i * n_real_pad * 16 + j * 256, dst + j * n_real_pad * 16 + i * 256, 1, 1, 1);
    	}
    }
    for(int a = 0; a < n_loop; a++)
    {
        vec_mul(ub_temp.get_ptr(0), dst + a * n_real_pad * 16 + a * 256, mask, 256);
        vec_mul(ub_tran.get_ptr(0), ub_temp.get_ptr(0), mask_dia, 256);
        pipe_barrier(PIPE_V);
        vec_trans(ub_tran.get_ptr(0), ub_tran.get_ptr(0), 1, 1, 1);
        vec_add(dst + a * n_real_pad * 16 + a * 256, ub_temp.get_ptr(0), ub_tran.get_ptr(0), 256);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_diag_gm2ub(__ub__ half *ub_buffer0,
                                                          __gm__ half *gm,
                                                          int64_t m_real_pad,
                                                          int64_t n_real_pad,
                                                          int64_t m_real,
                                                          int64_t n_real,
                                                          int64_t stride,
                                                          hablasDiagType_t diag,
                                                          hablasFillMode_t uplo)
{
    if (m_real % 16 || (stride - m_real) % 16)
    {
        for (int j = 0; j < n_real; ++j)
        {
            _memcpy(ub_buffer0 + j * m_real_pad, gm + j * stride, 1, m_real_pad / 16, 0, 0);
        }
    }
    else
    {
        _memcpy(ub_buffer0, gm, n_real, m_real / 16, 0, (stride - m_real) / 16);
    }
    set_flag(PIPE_MTE2, PIPE_S, 2);
    wait_flag(PIPE_MTE2, PIPE_S, 2);
    if (uplo == HABLAS_FILL_MODE_LOWER)
    {
        for (int col = 0; col < n_real; col++)
        {
            for (int row = 0; row < col; row++)
            {
                *(ub_buffer0 + col * m_real_pad + row) = 0.0;
            }
            if (diag == HABLAS_DIAG_UNIT)
            {
                *(ub_buffer0 + col * m_real_pad + col) = 1.0;
            }
        }
    }
    else
    {
        for (int col = 0; col < n_real; col++)
        {
            for (int row = col + 1; row < m_real; row++)
            {
                *(ub_buffer0 + col * m_real_pad + row) = 0.0;
            }
            if (diag == HABLAS_DIAG_UNIT)
            {
                *(ub_buffer0 + col * m_real_pad + col) = 1.0;
            }
        }
    }
}
#endif
