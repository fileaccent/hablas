#include "tools.h"

constexpr int UB_HALF_240KB = 240 * 1024 / 2;
constexpr int UB_HALF_8KB = 8 * 1024 / 2;

extern "C" __global__ __aicore__ void hablas_hsymv_kernel(hablasFillMode_t uplo, 
                                                          int64_t N, 
                                                          half alpha,
                                                          __gm__ half *matrixA,
                                                          int64_t lda, 
                                                          __gm__ half *x, 
                                                          int64_t incx, 
                                                          half beta, 
                                                          __gm__ half *y, 
                                                          int64_t incy)
{
    int64_t n = 176;
    int64_t vec_k = 16;
  
    Vector<float_8, L0C_MAX_SINGLE_SIZE / 8, HACL_L0C> output_y;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0A> input_x;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0B> input_A;

    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1x;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1A;

    Vector<half_16, UB_HALF_240KB / 16, HACL_UB> ub;

    __ub__ half *ubA1 = ub.get_ptr(0);
    __ub__ half *ubA2 = ubA1 + UB_HALF_64KB;
    __ub__ half *uby = ubA2 + UB_HALF_64KB;
    __ub__ half *ubx = uby + UB_HALF_8KB;

    __ub__ half *ub_temp1 = ubx + UB_HALF_8KB;

    int64_t n_tiles  = (N + n - 1) / n;
    int64_t n_remain = N % n;
    int64_t tiles_per_core;

    tiles_per_core = n_tiles / block_num;
    
    if (block_idx < n_tiles % block_num) 
    {
        tiles_per_core += 1;
    } 

    set_flag(PIPE_MTE3, PIPE_MTE2, 1);

    for (int64_t i = 0; i < tiles_per_core; i++) {
        int64_t col = (i * block_num + block_idx) % n_tiles;
        int64_t n_real = n;
        int64_t n_real_pad = n;
        if (col == n_tiles - 1 && n_remain > 0) {
            n_real = n_remain;
            n_real_pad = n_real % 16 ? (n_real & 0xFFFFFFF0) + 16 : n_real;
        }
        __gm__ half *Y_ptr = y + col * n * incy;
        wait_flag(PIPE_MTE3, PIPE_MTE2, 1);
        hablas_load_Vector_gm2ub(uby, Y_ptr, ub_temp1, n_real, incy);
        
        set_flag(PIPE_MTE2, PIPE_V, 2);
        wait_flag(PIPE_MTE2, PIPE_V, 2);
        vec_muls(uby, uby, beta, n_real_pad);

        set_flag(PIPE_V, PIPE_S, 2);
        wait_flag(PIPE_V, PIPE_S, 2);
        _memcpy(output_y.get_ptr(0), uby, 1, n_real_pad / 16, 0, 0, block_t::VECTOR);

        set_flag(PIPE_M, PIPE_MTE1, 0);

        set_flag(PIPE_V, PIPE_MTE2, 0);
        set_flag(PIPE_MTE1, PIPE_MTE3, 0);
        set_flag(PIPE_MTE3, PIPE_V, 0);
        set_flag(PIPE_V, PIPE_MTE2, 1);
        set_flag(PIPE_MTE1, PIPE_MTE3, 1);
        set_flag(PIPE_MTE3, PIPE_V, 1);

        set_flag(PIPE_V, PIPE_MTE2, 2);
        wait_flag(PIPE_V, PIPE_MTE2, 2);
        for (int64_t j = 0; j < n_tiles; j++)
        {
            int64_t k_real = n;
            int64_t k_real_pad = n;
            if (j == n_tiles - 1 && n_remain > 0) {
                k_real = n_remain;
                k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
            }
            else if(j != n_tiles - 1 && col == n_tiles - 1){
                k_real = n;
            }

            wait_flag(PIPE_M, PIPE_MTE1, 0);

            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
            wait_flag(PIPE_MTE3, PIPE_V, 0);
            __gm__ half *x_ptr = x + j * n * incx;
            wait_flag(PIPE_V, PIPE_MTE2, 0);
            hablas_load_Vector_gm2ub(ubx, x_ptr, ub_temp1, k_real, incx);

            set_flag(PIPE_V, PIPE_MTE2, 0);
            set_flag(PIPE_MTE2, PIPE_MTE3, 0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, 0);
            _memcpy(L1x.get_ptr(0), ubx, 1, k_real_pad / 16, 0, 0);

            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
            load2d(input_x.get_ptr(0), L1x.get_ptr(0), 0, (k_real_pad / 256) + 1, 1, false);
           
            set_flag(PIPE_MTE3, PIPE_V, 0);
            set_flag(PIPE_MTE1, PIPE_MTE3, 0);

            wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
            wait_flag(PIPE_MTE3, PIPE_V, 1);

            __gm__ half *A_ptr;

            if(uplo == HABLAS_FILL_MODE_LOWER)
            {
                if(j > col)
                {
                    A_ptr = matrixA + n * lda * col + j * n;
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real_pad, n_real_pad, k_real, n_real, lda);
                    set_flag(PIPE_MTE2, PIPE_V, 1);
                    wait_flag(PIPE_MTE2, PIPE_V, 1);
                    hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, n_real_pad, alpha);
                    set_flag(PIPE_V, PIPE_MTE2, 1);
                    set_flag(PIPE_V, PIPE_MTE3, 1);
                    wait_flag(PIPE_V, PIPE_MTE3, 1);
                    _memcpy(L1A.get_ptr(0), ubA2, 1, k_real_pad * n_real_pad / 16, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    load2d(input_A.get_ptr(0), L1A.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);
                }
                else if(j == col)
                {
                    A_ptr = matrixA + n * lda * col + j * n;
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real_pad, n_real_pad, k_real, n_real, lda);
                    set_flag(PIPE_MTE2, PIPE_V, 1);
                    wait_flag(PIPE_MTE2, PIPE_V, 1);
                    hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, n_real_pad, alpha);
                    set_flag(PIPE_V, PIPE_S, 1);
                    wait_flag(PIPE_V, PIPE_S, 1);
                    hablas_load_Matrix_dig_lower(ubA2, n_real_pad);
                    set_flag(PIPE_V, PIPE_MTE2, 1);
                    set_flag(PIPE_V, PIPE_MTE3, 1);
                    wait_flag(PIPE_V, PIPE_MTE3, 1);
                    _memcpy(L1A.get_ptr(0), ubA2, 1, k_real_pad * n_real_pad / 16, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    for (int i = 0; i < n_real_pad / 16; ++i)
                    {
                        load2d(input_A.get_ptr(0) + i * n_real_pad * 16, L1A.get_ptr(0), i * (n_real_pad / 16), i, 1, false);
                        load2d(input_A.get_ptr(0) + i * n_real_pad * 16 + i * 256, L1A.get_ptr(0), i * (n_real_pad / 16) + i, (n_real_pad / 16) - i, n_real_pad / 16, true);
                    }
                }
                else
                {
                    A_ptr = matrixA + n * lda * j + col * n;
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    hablas_load_matrix_gm2ub(ubA1, A_ptr, n_real_pad, k_real_pad, n_real, k_real, lda);
                    set_flag(PIPE_MTE2, PIPE_V, 1);
                    wait_flag(PIPE_MTE2, PIPE_V, 1);
                    hablas_load_input_matrix_ND2zZ(ubA2, ubA1, n_real_pad, k_real_pad, alpha);
                    set_flag(PIPE_V, PIPE_MTE2, 1);
                    set_flag(PIPE_V, PIPE_MTE3, 1);
                    wait_flag(PIPE_V, PIPE_MTE3, 1);
                    _memcpy(L1A.get_ptr(0), ubA2, 1, k_real_pad * n_real_pad / 16, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    for (int i = 0; i < k_real_pad / 16; ++i)
                    {
                        load2d(input_A.get_ptr(0) + i * n_real_pad * 16, L1A.get_ptr(0), i, n_real_pad / 16, k_real_pad / 16, true);
                    }
                }
            }
            else if(uplo == HABLAS_FILL_MODE_UPPER)
            {
                if(j < col)
                {
                    A_ptr = matrixA + n * lda * col + j * n;
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real_pad, n_real_pad, k_real, n_real, lda);
                    set_flag(PIPE_MTE2, PIPE_V, 1);
                    wait_flag(PIPE_MTE2, PIPE_V, 1);
                    hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, n_real_pad, alpha);
                    set_flag(PIPE_V, PIPE_MTE2, 1);
                    set_flag(PIPE_V, PIPE_MTE3, 1);
                    wait_flag(PIPE_V, PIPE_MTE3, 1);
                    _memcpy(L1A.get_ptr(0), ubA2, 1, k_real_pad * n_real_pad / 16, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    load2d(input_A.get_ptr(0), L1A.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);
                }
                else if(j == col)
                {
                    A_ptr = matrixA + n * lda * col + j * n;
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real_pad, n_real_pad, k_real, n_real, lda);
                    set_flag(PIPE_MTE2, PIPE_V, 1);
                    wait_flag(PIPE_MTE2, PIPE_V, 1);
                    hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, n_real_pad, alpha);
                    set_flag(PIPE_V, PIPE_S, 1);
                    wait_flag(PIPE_V, PIPE_S, 1);
                    hablas_load_Matrix_dig_upper(ubA2, n_real_pad);
                    set_flag(PIPE_V, PIPE_MTE2, 1);
                    set_flag(PIPE_V, PIPE_MTE3, 1);
                    wait_flag(PIPE_V, PIPE_MTE3, 1);
                    _memcpy(L1A.get_ptr(0), ubA2, 1, k_real_pad * n_real_pad / 16, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    for (int i = 0; i < n_real_pad / 16; ++i)
                    {
                        load2d(input_A.get_ptr(0) + i * n_real_pad * 16, L1A.get_ptr(0), i, i, n_real_pad / 16, true);
                        load2d(input_A.get_ptr(0) + i * n_real_pad * 16 + i * 256, L1A.get_ptr(0), i * (n_real_pad / 16) + i, (n_real_pad / 16) - i, 1, false);
                    }
                }
                else
                {
                    A_ptr = matrixA + n * lda * j + col * n;
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    hablas_load_matrix_gm2ub(ubA1, A_ptr, n_real_pad, k_real_pad, n_real, k_real, lda);
                    set_flag(PIPE_MTE2, PIPE_V, 1);
                    wait_flag(PIPE_MTE2, PIPE_V, 1);
                    hablas_load_input_matrix_ND2zZ(ubA2, ubA1, n_real_pad, k_real_pad, alpha);
                    set_flag(PIPE_V, PIPE_MTE2, 1);
                    set_flag(PIPE_V, PIPE_MTE3, 1);
                    wait_flag(PIPE_V, PIPE_MTE3, 1);
                    _memcpy(L1A.get_ptr(0), ubA2, 1, k_real_pad * n_real_pad / 16, 0, 0);
                    set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                    for (int i = 0; i < k_real_pad / 16; ++i)
                    {
                        load2d(input_A.get_ptr(0) + i * n_real_pad * 16, L1A.get_ptr(0), i, n_real_pad / 16, k_real_pad / 16, true);
                    }
                }
            }

            set_flag(PIPE_MTE3, PIPE_V, 1);
            set_flag(PIPE_MTE1, PIPE_MTE3, 1);

            set_flag(PIPE_MTE1, PIPE_M, 0);
            wait_flag(PIPE_MTE1, PIPE_M, 0);

            mmad(output_y.get_ptr(0), input_x.get_ptr(0), input_A.get_ptr(0), 1, k_real, n_real, 0);
            
            set_flag(PIPE_M, PIPE_MTE1, 0);
        }

        wait_flag(PIPE_M, PIPE_MTE1, 0);

        wait_flag(PIPE_V, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 0);

        wait_flag(PIPE_V, PIPE_MTE2, 1);
        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
        wait_flag(PIPE_MTE3, PIPE_V, 1);

        set_flag(PIPE_M, PIPE_S, 0);
        wait_flag(PIPE_M, PIPE_S, 0);

        _memcpy(uby, output_y.get_ptr(0), 1, n_real_pad / 16, 0, 0, block_t::VECTOR);
        set_flag(PIPE_S, PIPE_MTE3, 3);
        wait_flag(PIPE_S, PIPE_MTE3, 3);

        hablas_store_Vector_ub2gm(Y_ptr, uby, ub_temp1, n_real_pad, n_real,incy);

        set_flag(PIPE_MTE3, PIPE_MTE2, 1);
    }

    wait_flag(PIPE_MTE3, PIPE_MTE2, 1);
}
