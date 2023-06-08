//#pragma once
#include "tools.h"

#define T float
#define vec_T float_8
#define vec_N 8
#define matrix_T float_16x16
#define matrix_N 16
#define ptr_type float
// #define max_num 1
#define max_num 16 * 4
#define debug false

constexpr int UB_MAX_FLOAT_SIZE = 62 * 1024;
constexpr int UB_FLOAT_64KB = 32 * 1024 / 8;

HACL_INLINE __aicore__ void hablas_load_matrix_gmToL1(__l1__ float *L1, __gm__ float *gm, int64_t m_pad, int64_t n_pad, int64_t m, int64_t n, int64_t lda) {
    for (int i = 0; i < n; i++) {
        _memcpy(L1 + i * m_pad, gm + i * lda, 1, (m + 8 - 1) / 8, 0, 0);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_L1ToUb(__ub__ float* ub, __l1__ float *L1, int64_t m_pad, int64_t n_pad, int64_t m, int64_t n) {
    _memcpy(ub, L1, 1, m_pad * n_pad / 8, 0, 0);
}

HACL_INLINE __aicore__ void mmad_sgemv(__ub__ float *ub_A, __ub__ float *ub_X, __ub__ float *ub_Y, __ub__ float *ub_buffer0, int64_t m_pad, int64_t n_pad, int64_t m, int64_t n, hablasOperation_t trans) {
    if (trans == HABLAS_OP_N) {
        for (int i = 0; i < n; i++) {
            vec_axpy(ub_Y, ub_A + i * m_pad, *(ub_X + i), m);
        }
    } else {
        int m_loop = (m + 64 - 1) / 64;
        int m_remain = m % 64;
        for (int i = 0; i < m_loop; i++) {
            int m_real = 64;
            if (i == m_loop - 1 && m_remain > 0) {
                m_real = m_remain;
            }
            __hacl_details__::__hacl_intrinsic_move_mask(m_real);
            if (i == 0) {
                __hacl_details__::__hacl_intrinsic_vec_mul(
                    ub_A, 
                    ub_X + i * 64, 
                    ub_A + i * 64, 
                    n, 
                    m_pad / 8, 0, m_pad / 8,
                    1, 1, 1);
            } else {
                __hacl_details__::__hacl_intrinsic_vec_mla(
                    ub_A, 
                    ub_X + i * 64, 
                    ub_A + i * 64, 
                    n, 
                    m_pad / 8, 0, m_pad / 8,
                    1, 1, 1);
            }
        }
        __hacl_details__::__hacl_intrinsic_vec_reduce_add(ub_buffer0, ub_A, n, m_pad / 8, 1);
        vec_add(ub_Y, ub_Y, ub_buffer0, n);
    }
}


extern "C" __global__ __aicore__ void hablas_sgemv_kernel(
	hablasOperation_t trans,
	int64_t M,
	int64_t N,
	T alpha,
	__gm__ T *d_A,
	int64_t lda,
	__gm__ T *d_X,
	int64_t incx,
	T beta,
	__gm__ T *d_Y,
	int64_t incy
){
    int64_t m;
    int64_t n;
    if (trans == HABLAS_OP_N) {
       m = 512;
       n = 16;
    } else {
       m = 128;
       n = 64; // n 不能大于等于256
    }
    Vector<vec_T, UB_MAX_FLOAT_SIZE / 8, HACL_UB> ub;
    const int ub_buffer_size = (UB_MAX_FLOAT_SIZE - 2 * m * n - m - n - m - n) / 2;
    Vector<vec_T, 1024 * 32 / 8, HACL_L1> L1A1_L1;
    Vector<vec_T, 1024 * 32 / 8, HACL_L1> L1A2_L1;
    Vector<vec_T, 1024 * 32 / 8, HACL_L1> L1X_L1;
    __l1__ float *L1A1 = L1A1_L1.get_ptr(0);
    __l1__ float *L1A2 = L1A2_L1.get_ptr(0);
    __l1__ float *L1X = L1X_L1.get_ptr(0);
    __ub__ float *ubA1 = ub.get_ptr(0); 
    __ub__ float *ubA2 = ub.get_ptr(0) + m * n; 
    __ub__ float *ubX1 = ubA2 + m * n;
    __ub__ float *ubX2;
    __ub__ float *ubY;
    __ub__ float *ub_buffer0 = ubX1 + m + n + m + n;
    __ub__ float *ub_buffer1 = ub_buffer0 + ub_buffer_size;
    if (trans == HABLAS_OP_N) {
       ubX2 = ubX1 + n;
       ubY = ubX2 + n;
    } else {
       ubX2 = ubX1 + m;
       ubY = ubX2 + m;
    }
    int64_t m_tiles = (M + m - 1) / m;
    int64_t n_tiles = (N + n - 1) / n;

    int64_t m_remain = M % m;
    int64_t n_remain = N % n;
    
    int64_t m_real = m;
    int64_t n_real = n;
    int64_t m_real_pad = m;
    int64_t n_real_pad = n;
    int64_t tiles_num;
    if (trans == HABLAS_OP_N) {
       tiles_num = m_tiles;
    } else {
       tiles_num = n_tiles;
    }
    int64_t tiles_per_core = tiles_num / block_num;
    if (block_idx < tiles_num % block_num) ++tiles_per_core;
    int64_t x_size;
    int64_t y_size;
    int64_t x_tiles;
    int64_t x_real;
    int64_t y_real;
    int64_t x_real_pad;
    int64_t y_real_pad;

    if (trans == HABLAS_OP_N) {
        x_size = n;
        y_size = m;
        x_tiles = n_tiles;
        x_real = n;
        y_real = m;
        x_real_pad = n;
        y_real_pad = m;
    } else {
        x_size = m;
        y_size = n;
        x_tiles = m_tiles;
        x_real = m;
        y_real = n;
        x_real_pad = m;
        y_real_pad = n;
    }

    int inputX_size = ub_buffer_size / x_size * x_size;
    const int x_tile_num = inputX_size / x_size;
    set_flag(PIPE_MTE3, PIPE_MTE2, 3); // 用于控制当UB将y存储完毕再重新使用
    for (int64_t i = 0; i < tiles_per_core; ++i) {
        int64_t id = i * block_num + block_idx;
        if (trans == HABLAS_OP_N) {
            m_real = m;
            if (id == m_tiles - 1 && m_remain > 0) {
                m_real = m_remain;
            }
            y_real = m_real;
            if (m_real < m && m_real % 8 != 0) {
                m_real_pad = (m_real + 8 - 1) / 8 * 8;
            } else {
                m_real_pad = m_real;
            }
            y_real_pad = m_real_pad;
        } else {
            n_real = n;
            if (id == n_tiles - 1 && n_remain > 0) {
                n_real = n_remain;
            }
            y_real = n_real;
            if (n_real < n && n_real % 8 != 0) {
                n_real_pad = (n_real + 8 - 1) / 8 * 8;
            } else {
                n_real_pad = n_real;
            }
            y_real_pad = n_real_pad;
        }
        __gm__ float *Y_ptr = d_Y + id * y_size * incy;
        wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
        if (debug)pipe_barrier(PIPE_ALL);
        {
            if (incy == 1) {
                _memcpy(ubY, Y_ptr, y_real_pad);
                set_flag(PIPE_MTE2, PIPE_V, 3);
                wait_flag(PIPE_MTE2, PIPE_V, 3);
            } else {
                int64_t load_data_num = (ub_buffer_size) / incy * incy;
                if (load_data_num == 0) load_data_num++;
                int64_t loop = y_real_pad * incy / load_data_num;
                int64_t remain = y_real_pad * incy % load_data_num;
                set_flag(PIPE_S, PIPE_MTE2, 3);
                for (int loop_index = 0; loop_index < loop; loop_index++) {
                    wait_flag(PIPE_S, PIPE_MTE2, 3);
                    _memcpy(ub_buffer1, Y_ptr + loop_index * load_data_num, load_data_num);
                    set_flag(PIPE_MTE2, PIPE_S, 3);
                    wait_flag(PIPE_MTE2, PIPE_S, 3);
                    for (int i = 0; i < load_data_num / incy; i++) {
                        ubY[loop_index * load_data_num / incy + i] = ub_buffer1[i * incy];
                    }
                    set_flag(PIPE_S, PIPE_MTE2, 3);
                }
                wait_flag(PIPE_S, PIPE_MTE2, 3);
                if (remain > 0) {
                    _memcpy(ub_buffer1, Y_ptr + loop * load_data_num, remain);
                    set_flag(PIPE_MTE2, PIPE_S, 3);
                    wait_flag(PIPE_MTE2, PIPE_S, 3);
                    for (int i = 0; i < remain / incy; i++) {
                        ubY[loop * load_data_num / incy + i] = ub_buffer1[i * incy];
                    }
                }
                set_flag(PIPE_S, PIPE_V, 3);
                wait_flag(PIPE_S, PIPE_V, 3);
            }
            vec_muls(ubY, ubY, beta, y_real_pad);
        }
        if (debug) pipe_barrier(PIPE_ALL);
        set_flag(PIPE_V, PIPE_MTE2, 2);
        int64_t load_x_total = (x_tiles + x_tile_num - 1) / x_tile_num;
        for (int x_block_index = 0; x_block_index < load_x_total; x_block_index++) {
            __gm__ float *X_ptr = d_X + x_block_index * x_tile_num * x_size * incx;
            int64_t x_tile_loop = x_tile_num;
            if (x_block_index == load_x_total - 1 && x_tiles % x_tile_num > 0) {
                x_tile_loop = x_tiles % x_tile_num;
            }
            int64_t x_block_size = x_size * x_tile_loop;
            int64_t x_block_size_pad  = (x_block_size + 8 - 1) / 8 * 8;
            wait_flag(PIPE_V, PIPE_MTE2, 2);
            if (debug) pipe_barrier(PIPE_ALL);
            {
                if (incx == 1) {
                    _memcpy(ub_buffer1, X_ptr, x_block_size_pad);
                    set_flag(PIPE_MTE2, PIPE_V, 2);
                    wait_flag(PIPE_MTE2, PIPE_V, 2);
                } else {
                    int64_t load_data_num = (ub_buffer_size) / incx * incx;
                    if (load_data_num == 0) load_data_num++;
                    int64_t loop = (x_block_size * incx - incx + 1) / load_data_num;
                    int64_t remain = (x_block_size * incx - incx + 1) % load_data_num;
                    set_flag(PIPE_S, PIPE_MTE2, 3);
                    for (int loop_index = 0; loop_index < loop; loop_index++) {
                        wait_flag(PIPE_S, PIPE_MTE2, 3);
                        _memcpy(ub_buffer0, X_ptr + loop_index * load_data_num, load_data_num);
                        set_flag(PIPE_MTE2, PIPE_S, 2);
                        wait_flag(PIPE_MTE2, PIPE_S, 2);
                        for (int i = 0; i < load_data_num / incx; i++) {
                            ub_buffer1[loop_index * load_data_num / incx + i] = ub_buffer0[i * incx];
                        }
                        set_flag(PIPE_S, PIPE_MTE2, 3);
                    }
                    wait_flag(PIPE_S, PIPE_MTE2, 3);
                    if (remain > 0) {
                        _memcpy(ub_buffer0, X_ptr + loop * load_data_num, remain);
                        set_flag(PIPE_MTE2, PIPE_S, 2);
                        wait_flag(PIPE_MTE2, PIPE_S, 2);
                        for (int i = 0; i < remain / incx; i++) {
                            ub_buffer1[loop * load_data_num / incx + i] = ub_buffer0[i * incx];
                        }
                    }
                    set_flag(PIPE_S, PIPE_V, 2);
                    wait_flag(PIPE_S, PIPE_V, 2);
                }
                vec_muls(ub_buffer1, ub_buffer1, alpha, x_block_size_pad);
                // 防止ubB数据踩踏
                set_flag(PIPE_V, PIPE_MTE3, 2);
                wait_flag(PIPE_V, PIPE_MTE3, 2);
                _memcpy(L1X, ub_buffer1, x_block_size_pad / 8, 1, 0, 0);
            }
            if (debug) pipe_barrier(PIPE_ALL);
            // A 矩阵执行双缓存
            set_flag(PIPE_MTE1, PIPE_MTE2, 0); // L1A1
            set_flag(PIPE_MTE1, PIPE_MTE2, 1); // L1A2

            set_flag(PIPE_V, PIPE_MTE1, 0); // ubA1, ubX1
            set_flag(PIPE_V, PIPE_MTE1, 1); // ubA2, ubX2
            set_flag(PIPE_V, PIPE_MTE1, 2); // ubY
            for (int x_tile_index = 0; x_tile_index < x_tile_loop; x_tile_index += 2) {
                const int64_t j = x_block_index * x_tile_num + x_tile_index;
                if (trans == HABLAS_OP_N) {
                    n_real = n;
                    if (j == n_tiles - 1 && n_remain > 0)
                    {
                        n_real = n_remain;
                    }
                    x_real = n_real;
                    if (n_real < n && n_real % 8 != 0)
                    {
                        n_real_pad = (n_real + 8 - 1) / 8 * 8;
                    }
                    else
                    {
                        n_real_pad = n_real;
                    }
                    x_real_pad = n_real_pad;
                } else {
                    m_real = m;
                    if (j == m_tiles - 1 && m_remain > 0)
                    {
                        m_real = m_remain;
                    }
                    x_real = m_real;
                    if (m_real < m && m_real % 8 != 0)
                    {
                        m_real_pad = (m_real + 8 - 1) / 8 * 8;
                    }
                    else
                    {
                        m_real_pad = m_real;
                    }
                    x_real_pad = m_real_pad;
                }
                __gm__ float *A1_ptr;
                if (trans == HABLAS_OP_N) {
                    A1_ptr = d_A + j * n * lda + id * m;
                } else {
                    A1_ptr = d_A + id * n * lda + j * m;
                }
                
                // 控制L1A1的使用
                wait_flag(PIPE_MTE1, PIPE_MTE2, 0);
                if (debug) pipe_barrier(PIPE_ALL);
                hablas_load_matrix_gmToL1(L1A1, A1_ptr, m, n, m_real, n_real, lda);
                set_flag(PIPE_MTE2, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, 0);

                // 控制ubA的使用
                wait_flag(PIPE_V, PIPE_MTE1, 0);
                // wait_flag(PIPE_S, PIPE_MTE1, 0);
                if (debug) pipe_barrier(PIPE_ALL);
                hablas_load_matrix_L1ToUb(ubA1, L1A1, m, n, m_real, n_real);
                
                set_flag(PIPE_MTE1, PIPE_MTE2, 0);
                if (x_tile_index == 0) {
                    set_flag(PIPE_MTE3, PIPE_MTE1, 2);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 2);
                }

                // 控制ubX1的使用
                if (debug) pipe_barrier(PIPE_ALL);
                hablas_load_matrix_L1ToUb(ubX1, L1X + x_tile_index * x_size, x_real_pad, 1, x_real, 1);

                set_flag(PIPE_MTE1, PIPE_V, 0);
                wait_flag(PIPE_MTE1, PIPE_V, 0);
                set_flag(PIPE_MTE1, PIPE_S, 0);
                wait_flag(PIPE_MTE1, PIPE_S, 0);

                // 控制ubY的使用
                wait_flag(PIPE_V, PIPE_MTE1, 2);
                if (debug) pipe_barrier(PIPE_ALL);
                mmad_sgemv(ubA1, ubX1, ubY, ub_buffer0, m, n, m_real, n_real, trans);

                set_flag(PIPE_V, PIPE_MTE1, 0);
                set_flag(PIPE_V, PIPE_MTE1, 2);
            

                if (x_tile_index + 1 >= x_tile_loop) break;

                if (trans == HABLAS_OP_N) {
                    n_real = n;
                    if (j + 1 == n_tiles - 1 && n_remain > 0)
                    {
                        n_real = n_remain;
                    }
                    x_real = n_real;
                    if (n_real < n && n_real % 8 != 0)
                    {
                        n_real_pad = (n_real + 8 - 1) / 8 * 8;
                    }
                    else
                    {
                        n_real_pad = n_real;
                    }
                    x_real_pad = n_real_pad;
                } else {
                    m_real = m;
                    if (j + 1 == m_tiles - 1 && m_remain > 0)
                    {
                        m_real = m_remain;
                    }
                    x_real = m_real;
                    if (m_real < m && m_real % 8 != 0)
                    {
                        m_real_pad = (m_real + 8 - 1) / 8 * 8;
                    }
                    else
                    {
                        m_real_pad = m_real;
                    }
                    x_real_pad = m_real_pad;
                }
                __gm__ float* A2_ptr;
                if (trans == HABLAS_OP_N) {
                    A2_ptr = d_A + (j + 1) * n * lda + id * m;
                } else {
                    A2_ptr = d_A + id * n * lda + (j + 1) * m;
                }
                
                // 控制L1A2的使用
                wait_flag(PIPE_MTE1, PIPE_MTE2, 1);
                if (debug) pipe_barrier(PIPE_ALL);
                hablas_load_matrix_gmToL1(L1A2, A2_ptr, m, n, m_real, n_real, lda);
                set_flag(PIPE_MTE2, PIPE_MTE1, 1);
                wait_flag(PIPE_MTE2, PIPE_MTE1, 1);

                // 控制ubA的使用
                wait_flag(PIPE_V, PIPE_MTE1, 1);
                if (debug) pipe_barrier(PIPE_ALL);
                hablas_load_matrix_L1ToUb(ubA2, L1A2, m, n, m_real, n_real);
                
                set_flag(PIPE_MTE1, PIPE_MTE2, 1);
                // 控制ubX2的使用
                if (debug) pipe_barrier(PIPE_ALL);
                hablas_load_matrix_L1ToUb(ubX2, L1X + (x_tile_index + 1) * x_size, x_real_pad, 1, x_real, 1);
                
                set_flag(PIPE_MTE1, PIPE_V, 1);
                wait_flag(PIPE_MTE1, PIPE_V, 1);
                set_flag(PIPE_MTE1, PIPE_S, 1);
                wait_flag(PIPE_MTE1, PIPE_S, 1);

                // 控制ubY的使用
                wait_flag(PIPE_V, PIPE_MTE1, 2);
                if (debug) pipe_barrier(PIPE_ALL);
                mmad_sgemv(ubA2, ubX2, ubY, ub_buffer0, m, n, m_real, n_real, trans);
                if (debug) pipe_barrier(PIPE_ALL);
                set_flag(PIPE_V, PIPE_MTE1, 1);
                set_flag(PIPE_V, PIPE_MTE1, 2);
            }
            wait_flag(PIPE_MTE1, PIPE_MTE2, 0); // L1A1
            wait_flag(PIPE_MTE1, PIPE_MTE2, 1); // L1A2
            
            wait_flag(PIPE_V, PIPE_MTE1, 0); // ubA
            wait_flag(PIPE_V, PIPE_MTE1, 1); // ubX1
            wait_flag(PIPE_V, PIPE_MTE1, 2); // ubY

            set_flag(PIPE_V, PIPE_MTE2, 2);
        }
        wait_flag(PIPE_V, PIPE_MTE2, 2);
        if (debug) pipe_barrier(PIPE_ALL);
        {
            if (incy == 1) {
                set_flag(PIPE_V, PIPE_MTE3, 3);
                wait_flag(PIPE_V, PIPE_MTE3, 3);
                _memcpy(Y_ptr, ubY, y_real_pad);
            } else {
                set_flag(PIPE_V, PIPE_MTE2, 3);
                wait_flag(PIPE_V, PIPE_MTE2, 3);
                int64_t load_data_num = (ub_buffer_size) / incy * incy;
                if (load_data_num == 0) load_data_num++;
                int64_t loop = y_real_pad * incy / load_data_num;
                int64_t remain = y_real_pad * incy % load_data_num;
                set_flag(PIPE_MTE3, PIPE_MTE2, 3);
                for (int loop_index = 0; loop_index < loop; loop_index++) {
                    wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
                    _memcpy(ub_buffer1, Y_ptr + loop_index * load_data_num, load_data_num);
                    set_flag(PIPE_MTE2, PIPE_S, 3);
                    wait_flag(PIPE_MTE2, PIPE_S, 3);
                    for (int i = 0; i < load_data_num / incy; i++) {
                        ub_buffer1[i * incy] = ubY[loop_index * load_data_num / incy + i];
                    }
                    set_flag(PIPE_S, PIPE_MTE3, 3);
                    wait_flag(PIPE_S, PIPE_MTE3, 3);
                    _memcpy(Y_ptr + loop_index * load_data_num, ub_buffer1, load_data_num);
                    set_flag(PIPE_MTE3, PIPE_MTE2, 3);
                }
                wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
                if (remain > 0) {
                    _memcpy(ub_buffer1, Y_ptr + loop * load_data_num, remain);
                    set_flag(PIPE_MTE2, PIPE_S, 3);
                    wait_flag(PIPE_MTE2, PIPE_S, 3);
                    for (int i = 0; i < remain / incy; i++) {
                        ub_buffer1[i * incy] = ubY[loop * load_data_num / incy + i];
                    }
                    set_flag(PIPE_S, PIPE_MTE3, 3);
                    wait_flag(PIPE_S, PIPE_MTE3, 3);
                    _memcpy(Y_ptr + loop * load_data_num, ub_buffer1, remain);
                }
            }
        }
        if (debug) pipe_barrier(PIPE_ALL);
        set_flag(PIPE_MTE3, PIPE_MTE2, 3);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
}