#include "tools.h"

extern "C" __global__ __aicore__ void hablas_hgemv_kernel(
	hablasOperation_t trans,
	int64_t M,
	int64_t N,
	half alpha,
	__gm__ half *d_A,
	int64_t lda,
	__gm__ half *d_X,
	int64_t incx,
	half beta,
	__gm__ half *d_Y,
	int64_t incy
)
{
    Vector<half_16, UB_MAX_HALF_SIZE / 32, HACL_UB> ub;
    Vector<float_8, L0C_MAX_SINGLE_SIZE / 8, HACL_L0C> result;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 2 / 16, HACL_L0B> inputA1;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 2 / 16, HACL_L0B> inputA2;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0A> inputX;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1A;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1X;

    __ub__ half *ubA1 = ub.get_ptr(0);
    __ub__ half *ubA2 = ubA1 + UB_HALF_64KB;
    __ub__ half *ubB1 = ubA2 + UB_HALF_64KB;
    __ub__ half *ubB2 = ubB1 + UB_HALF_64KB;
    __ub__ half *ub_buffer0 = ub.get_ptr(0);
    __ub__ half *ub_buffer1 = ub.get_ptr(0) + 2 * UB_HALF_64KB;

    int64_t m;
    int64_t n;
    if (trans == HABLAS_OP_N) {
        m = 64;
        n = 256;
    } else {
        m = 256;
        n = 64;
    }

    int64_t m_tiles = (M + m - 1) / m;
    int64_t n_tiles = (N + n - 1) / n;

    int64_t m_remain = M % m;
    int64_t n_remain = N % n;
    
    int64_t m_real_pad = m;
    int64_t n_real_pad = n;
    int64_t m_real = m;
    int64_t n_real = n;
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

    int inputX_size = m * n;
    const int x_tile_num = inputX_size / x_size; // L0B可以容纳的X向量块的个数
    set_flag(PIPE_MTE3, PIPE_MTE2, 3); // 用于控制当UB将y存储完毕再重新使用
    for (int64_t i = 0; i < tiles_per_core; ++i) {
        int64_t id = i * block_num + block_idx;
        if (trans == HABLAS_OP_N) {
            m_real = m;
            if (id == m_tiles - 1 && m_remain > 0) {
                m_real = m_remain;
            }
            y_real = m_real;
            if (m_real < m && m_real % 16 != 0) {
                m_real_pad = (m_real & 0xFFFFFFF0) + 16;
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
            if (n_real < n && n_real % 16 != 0) {
                n_real_pad = (n_real & 0xFFFFFFF0) + 16;
            } else {
                n_real_pad = n_real;
            }
            y_real_pad = n_real_pad;
        }
        // prefetch C to ub_cs
        __gm__ half *Y_ptr = d_Y + id * y_size * incy;
        wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
        {
            if (incy == 1) {
                _memcpy(ub_buffer0, Y_ptr, y_real_pad);
                set_flag(PIPE_MTE2, PIPE_V, 3);
                wait_flag(PIPE_MTE2, PIPE_V, 3);
            } else {
                int64_t load_data_num = (2 * UB_HALF_64KB) / incy * incy;
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
                        ub_buffer0[loop_index * load_data_num / incy + i] = ub_buffer1[i * incy];
                    }
                    set_flag(PIPE_S, PIPE_MTE2, 3);
                }
                wait_flag(PIPE_S, PIPE_MTE2, 3);
                if (remain > 0) {
                    _memcpy(ub_buffer1, Y_ptr + loop * load_data_num, remain);
                    set_flag(PIPE_MTE2, PIPE_S, 3);
                    wait_flag(PIPE_MTE2, PIPE_S, 3);
                    for (int i = 0; i < remain / incy; i++) {
                        ub_buffer0[loop * load_data_num / incy + i] = ub_buffer1[i * incy];
                    }
                }
                set_flag(PIPE_S, PIPE_V, 3);
                wait_flag(PIPE_S, PIPE_V, 3);
            }

            vec_muls(ub_buffer0, ub_buffer0, beta, y_real_pad);
            pipe_barrier(PIPE_V);
            _memcpy(ub_buffer0, ub_buffer0, y_real_pad / 16, 1, 15, 0);
            set_flag(PIPE_V, PIPE_MTE3, 3);
            wait_flag(PIPE_V, PIPE_MTE3, 3);
            _memcpy(result.get_ptr(0), ub_buffer0, y_real_pad / 16, 1, 0, 0, block_t::MATRIX);
        }
        int64_t load_x_total = (x_tiles + x_tile_num - 1) / x_tile_num;
        for (int x_block_index = 0; x_block_index < load_x_total; x_block_index++) {
            // 假设为 m * n 的块，x每次读 m；如果要一起读能读n段, 一共需要读 m_tiles 次, 合并读取则需要 m_tiles / n
            // 在执行循环时, y要读取完毕
            if (x_block_index == 0) {
                set_flag(PIPE_V, PIPE_MTE2, 2);
                wait_flag(PIPE_V, PIPE_MTE2, 2);
            }
            __gm__ half *X_ptr = d_X + x_block_index * x_tile_num * x_size * incx;
            int64_t x_tile_loop = x_tile_num;
            if (x_block_index == load_x_total - 1 && x_tiles % x_tile_num > 0) {
                x_tile_loop = x_tiles % x_tile_num;
            }
            int64_t x_block_size = x_size * x_tile_loop;
            int64_t x_block_size_pad  = (x_block_size & 0xFFFFFFF0) + 16;
            
            set_flag(PIPE_M, PIPE_MTE2, 2);
            wait_flag(PIPE_M, PIPE_MTE2, 2);
            {
                if (incx == 1) {
                    _memcpy(ub_buffer1, X_ptr, x_block_size_pad);
                    set_flag(PIPE_MTE2, PIPE_V, 2);
                    wait_flag(PIPE_MTE2, PIPE_V, 2);
                } else {
                    int64_t load_data_num = (2 * UB_HALF_64KB) / incx * incx;
                    if (load_data_num == 0) load_data_num++;
                    int64_t loop = x_block_size_pad * incx / load_data_num;
                    int64_t remain = x_block_size_pad * incx % load_data_num;
                    set_flag(PIPE_S, PIPE_MTE2, 2);
                    for (int loop_index = 0; loop_index < loop; loop_index++) {
                        wait_flag(PIPE_S, PIPE_MTE2, 2);
                        _memcpy(ub_buffer0, X_ptr + loop_index * load_data_num, load_data_num);
                        set_flag(PIPE_MTE2, PIPE_S, 2);
                        wait_flag(PIPE_MTE2, PIPE_S, 2);
                        for (int i = 0; i < load_data_num / incx; i++) {
                            ub_buffer1[loop_index * load_data_num / incx + i] = ub_buffer0[i * incx];
                        }
                        set_flag(PIPE_S, PIPE_MTE2, 2);
                    }
                    wait_flag(PIPE_S, PIPE_MTE2, 2);
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

                set_flag(PIPE_V, PIPE_MTE3, 2);
                wait_flag(PIPE_V, PIPE_MTE3, 2);
                _memcpy(L1X.get_ptr(0), ub_buffer1, x_block_size_pad / 16, 1, 0, 0);
            }
            // 等待L1X读取完毕
            set_flag(PIPE_MTE3, PIPE_MTE1, 2);
            wait_flag(PIPE_MTE3, PIPE_MTE1, 2);
            load2d(inputX.get_ptr(0), L1X.get_ptr(0), 0, ((x_block_size + 16 * 16 - 1) / (16 * 16)), 1, false);
            set_flag(PIPE_MTE1, PIPE_M, 2); // 用于同步x和mmad
            // A 矩阵执行双缓存
            set_flag(PIPE_M, PIPE_MTE1, 0); // 当矩阵计算完毕再L1 -> L0A1
            set_flag(PIPE_M, PIPE_MTE1, 1); // 当矩阵计算完毕再L1 -> L0A2
            set_flag(PIPE_V, PIPE_MTE2, 0); // 同步ubA的使用, 防止踩踏
            set_flag(PIPE_V, PIPE_MTE2, 1); // 同步ubB的使用, 防止踩踏
            for (int x_tile_index = 0; x_tile_index < x_tile_loop; x_tile_index += 2) {
                const int64_t j = x_block_index * x_tile_num + x_tile_index;
                if (trans == HABLAS_OP_N) {
                    n_real = n;
                    if (j == n_tiles - 1 && n_remain > 0)
                    {
                        n_real = n_remain;
                    }
                    x_real = n_real;
                    if (n_real < n && n_real % 16 != 0)
                    {
                        n_real_pad = (n_real & 0xFFFFFFF0) + 16;
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
                    if (m_real < m && m_real % 16 != 0)
                    {
                        m_real_pad = (m_real & 0xFFFFFFF0) + 16;
                    }
                    else
                    {
                        m_real_pad = m_real;
                    }
                    x_real_pad = m_real_pad;
                }
                __gm__ half *A1_ptr;
                if (trans == HABLAS_OP_N) {
                    A1_ptr = d_A + j * n * lda + id * m;
                } else {
                    A1_ptr = d_A + id * n * lda + j * m;
                }
                wait_flag(PIPE_V, PIPE_MTE2, 0);
                hablas_load_matrix_gm2ub(ubA1, A1_ptr, m_real_pad, n_real_pad, m_real, n_real, lda);
                set_flag(PIPE_MTE2, PIPE_V, 0);
                wait_flag(PIPE_MTE2, PIPE_V, 0);
                hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, n_real_pad, (half)1.0);
                set_flag(PIPE_V, PIPE_MTE2, 0);
                set_flag(PIPE_V, PIPE_MTE3, 0);
                wait_flag(PIPE_V, PIPE_MTE3, 0);
                hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, n_real_pad);
                set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                wait_flag(PIPE_M, PIPE_MTE1, 0); // 等待L1A用完
                if (trans == HABLAS_OP_N) {
                    for (int i = 0; i < n_real_pad / 16; ++i)
                    {
                        load2d(inputA1.get_ptr(0) + i * m_real_pad * 16, L1A.get_ptr(0), i, m_real_pad / 16, n_real_pad / 16, true);
                    }
                } else {
                    load2d(inputA1.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (n_real_pad / 16), 1, false);
                }
                set_flag(PIPE_MTE1, PIPE_M, 0); // 用于同步A1和mmad
                if (x_tile_index == 0) {
                    wait_flag(PIPE_MTE1, PIPE_M, 2);
                }
                wait_flag(PIPE_MTE1, PIPE_M, 0);
                if (trans == HABLAS_OP_N) {
                    mmad(result.get_ptr(0), inputX.get_ptr(x_tile_index * x_size / 16), inputA1.get_ptr(0), 1, n_real, m_real, 0);
                } else {
                    mmad(result.get_ptr(0), inputX.get_ptr(x_tile_index * x_size / 16), inputA1.get_ptr(0), 1, m_real, n_real, 0);
                }
                set_flag(PIPE_M, PIPE_MTE1, 0); // 计算完inputA1再读取
                
                pipe_barrier(PIPE_M);
                set_flag(PIPE_M, PIPE_S, 2);
                wait_flag(PIPE_M, PIPE_S, 2);

                if (x_tile_index + 1 >= x_tile_loop) break;
                if (trans == HABLAS_OP_N) {
                    n_real = n;
                    if (j + 1 == n_tiles - 1 && n_remain > 0)
                    {
                        n_real = n_remain;
                    }
                    x_real = n_real;
                    if (n_real < n && n_real % 16 != 0)
                    {
                        n_real_pad = (n_real & 0xFFFFFFF0) + 16;
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
                    if (m_real < m && m_real % 16 != 0)
                    {
                        m_real_pad = (m_real & 0xFFFFFFF0) + 16;
                    }
                    else
                    {
                        m_real_pad = m_real;
                    }
                    x_real_pad = m_real_pad;
                }
                __gm__ half* A2_ptr;
                if (trans == HABLAS_OP_N) {
                    A2_ptr = d_A + (j + 1) * n * lda + id * m;
                } else {
                    A2_ptr = d_A + id * n * lda + (j + 1) * m;
                }
                wait_flag(PIPE_V, PIPE_MTE2, 1);
                hablas_load_matrix_gm2ub(ubB1, A2_ptr, m_real_pad, n_real_pad, m_real, n_real, lda);
                set_flag(PIPE_MTE2, PIPE_V, 1);
                wait_flag(PIPE_MTE2, PIPE_V, 1);
                hablas_load_input_matrix_ND2zZ(ubB2, ubB1, m_real_pad, n_real_pad, (half)1.0);
                set_flag(PIPE_V, PIPE_MTE2, 1);
                set_flag(PIPE_V, PIPE_MTE3, 1);
                wait_flag(PIPE_V, PIPE_MTE3, 1);
                hablas_load_input_matrix_ub2l1(L1X.get_ptr(0), ubB2, m_real_pad, n_real_pad);
                set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                wait_flag(PIPE_M, PIPE_MTE1, 1);
                if (trans == HABLAS_OP_N) {
                    for (int i = 0; i < n_real_pad / 16; ++i)
                    {
                        load2d(inputA2.get_ptr(0) + i * m_real_pad * 16, L1X.get_ptr(0), i, m_real_pad / 16, n_real_pad / 16, true);
                    }
                } else {
                    load2d(inputA2.get_ptr(0), L1X.get_ptr(0), 0, (m_real_pad / 16) * (n_real_pad / 16), 1, false);
                }
                set_flag(PIPE_MTE1, PIPE_M, 1); // 用于同步A1和mmad
                wait_flag(PIPE_MTE1, PIPE_M, 1);
                if (trans == HABLAS_OP_N) {
                    mmad(result.get_ptr(0), inputX.get_ptr((x_tile_index + 1) * x_size / 16), inputA2.get_ptr(0), 1, n_real, m_real, 0);
                } else {
                    mmad(result.get_ptr(0), inputX.get_ptr((x_tile_index + 1) * x_size / 16), inputA2.get_ptr(0), 1, m_real, n_real, 0);
                }
                pipe_barrier(PIPE_M);
                set_flag(PIPE_M, PIPE_MTE1, 1); // 计算完inputA2再读取
            }
            wait_flag(PIPE_V, PIPE_MTE2, 0);
            wait_flag(PIPE_V, PIPE_MTE2, 1);
            wait_flag(PIPE_M, PIPE_MTE1, 0);
            wait_flag(PIPE_M, PIPE_MTE1, 1);
        }
        set_flag(PIPE_M, PIPE_V, 3);
        wait_flag(PIPE_M, PIPE_V, 3);
        set_flag(PIPE_M, PIPE_S, 3);
        wait_flag(PIPE_M, PIPE_S, 3);
        set_flag(PIPE_M, PIPE_MTE1, 3);
        wait_flag(PIPE_M, PIPE_MTE1, 3);
        {
            _memcpy(ub_buffer0, result.get_ptr(0), 1, y_real_pad / 16, 0, 0, block_t::MATRIX);
            pipe_barrier(PIPE_V);
            _memcpy(ub_buffer0, ub_buffer0, y_real_pad / 16, 1, 0, 15);
            if (incy == 1) {
                set_flag(PIPE_V, PIPE_MTE3, 3);
                wait_flag(PIPE_V, PIPE_MTE3, 3);
                _memcpy(Y_ptr, ub_buffer0, y_real_pad);
            } else {
                set_flag(PIPE_V, PIPE_MTE2, 3);
                wait_flag(PIPE_V, PIPE_MTE2, 3);
                int64_t load_data_num = (2 * UB_HALF_64KB) / incy * incy;
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
                        ub_buffer1[i * incy] = ub_buffer0[loop_index * load_data_num / incy + i];
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
                        ub_buffer1[i * incy] = ub_buffer0[loop * load_data_num / incy + i];
                    }
                    set_flag(PIPE_S, PIPE_MTE3, 3);
                    wait_flag(PIPE_S, PIPE_MTE3, 3);
                    _memcpy(Y_ptr + loop * load_data_num, ub_buffer1, remain);
                }
            }
        }
        set_flag(PIPE_MTE3, PIPE_MTE2, 3);
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, 3);
}