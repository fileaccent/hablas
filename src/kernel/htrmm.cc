#include "tools.h"

#ifndef CAMODEL_PROFILING
extern "C" __global__ __aicore__ void hablas_htrmm_kernel(hablasSideMode_t side,
                                                          hablasFillMode_t uplo,
                                                          hablasOperation_t transA,
                                                          hablasDiagType_t diag,
                                                          int64_t M,
                                                          int64_t N,
                                                          half alpha,
                                                          __gm__ half *matrixA,
                                                          int64_t lda,
                                                          __gm__ half *matrixB,
                                                          int64_t ldb,
                                                          __gm__ half *matrixC,
                                                          int64_t ldc)
#else
extern "C" __global__ __aicore__ void hablas_htrmm_kernel(__gm__ half *matrixA,
                                                          __gm__ half *matrixB,
                                                          __gm__ half *matrixC)
#endif
{
    Vector<float_8, L0C_MAX_SINGLE_SIZE / 8, HACL_L0C> result;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0A> inputA;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0B> inputB;

    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1A;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1B;

    Vector<half_16, UB_MAX_HALF_SIZE / 16, HACL_UB> ub;

#ifdef CAMODEL_PROFILING
    hablasSideMode_t side= HABLAS_SIDE_LEFT;
    hablasFillMode_t uplo= HABLAS_FILL_MODE_UPPER;
    hablasOperation_t transA = HABLAS_OP_N;
    hablasDiagType_t diag = HABLAS_DIAG_NON_UNIT;
    int64_t M = 512;
    int64_t N = 512;
    half alpha = 1;
    int64_t lda = M;
    int64_t ldb = M;
    int64_t ldc = M;
#endif

    int64_t m = 144;
    int64_t n = 128;

    while (M % m < 16 && M % m > 0 && m > 0 && M >= 16) {
        m -= 16;
    }

    __ub__ half *ubA1 = ub.get_ptr(0);
    __ub__ half *ubA2 = ubA1 + UB_HALF_64KB;
    __ub__ half *ubB1 = ubA2 + UB_HALF_64KB;
    __ub__ half *ubB2 = ubB1 + UB_HALF_64KB;

    __ub__ half *ub_buffer0 = ub.get_ptr(0);
    __ub__ half *ub_buffer1 = ub_buffer0 + 2 * UB_HALF_64KB;

    if(side == HABLAS_SIDE_LEFT) {      
        int64_t m_tiles  = (M + m - 1) / m;
        int64_t n_tiles  = (N + n - 1) / n;
        int64_t k_loop = (M + m - 1) / m;

        int64_t m_remain = M % m;
        int64_t n_remain = N % n;
        int64_t k_remain = M % m;

        int64_t tiles_num = m_tiles * n_tiles;
        int64_t tiles_per_core = tiles_num / block_num;
        if (block_idx < tiles_num % block_num) {
             ++tiles_per_core;
        }

        set_flag(PIPE_MTE3, PIPE_MTE2, 1);

        for (int64_t i = 0; i < tiles_per_core; ++i) {
            int64_t id = i * block_num + block_idx;
            int64_t col = id / m_tiles;
            int64_t row = id % m_tiles;
            int64_t m_real = m;
            int64_t n_real = n;
            int64_t m_real_pad = m;
            int64_t n_real_pad = n;

            if (row == m_tiles - 1 && m_remain)
            {
                m_real = m_remain;
                m_real_pad = m_real % 16 ? (m_real & 0xFFFFFFF0) + 16 : m_real;
            }

            if (col == n_tiles - 1 && n_remain)
            {
                n_real = n_remain;
                n_real_pad = n_real % 16 ? (n_real & 0xFFFFFFF0) + 16 : n_real;
            }
            
            __gm__ half *C_ptr = matrixC + col * ldc * n +  row * m;
            wait_flag(PIPE_MTE3, PIPE_MTE2, 1);

            set_flag(PIPE_M, PIPE_MTE1, 0);
            set_flag(PIPE_V, PIPE_MTE2, 0);
            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
            set_flag(PIPE_MTE3, PIPE_V, 0);
            set_flag(PIPE_V, PIPE_MTE2, 1);
            set_flag(PIPE_MTE1, PIPE_MTE3, 1);
            set_flag(PIPE_MTE3, PIPE_V, 1);

            set_flag(PIPE_MTE3, PIPE_MTE2, 0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, 0);

            if (uplo == HABLAS_FILL_MODE_UPPER) {
                if (transA == HABLAS_OP_N) {
                    for (int j = row; j < k_loop; j++) 
                    {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                        }
                        __gm__ half *A_ptr;
                        __gm__ half *B_ptr;
                        A_ptr = matrixA + j * m * lda + row * m;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == row) {
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_diag_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, lda, diag, uplo);
                            
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {

                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, lda);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }

                        B_ptr = matrixB + col * n * ldb + j * m;
                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);

                        hablas_load_matrix_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, ldb);
                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);

                        hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, alpha);
                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);

                        hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);
                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);

                        load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);

                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1);   

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == row) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real_pad, k_real, n_real_pad, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real_pad, k_real, n_real_pad, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);
                    }
                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);

                } 
                else {    //transA == HABLAS_OP_T
                    for (int j = 0; j < row + 1; j++) 
                    {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                        }
                        __gm__ half *A_ptr;
                        __gm__ half *B_ptr;
                        A_ptr = matrixA + j * m + row * m * lda;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == row) {
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);
                            hablas_load_matrix_diag_gm2ub(ubA1, A_ptr, k_real_pad, m_real_pad, k_real, m_real, lda, diag, uplo);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, m_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, k_real_pad, m_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < m_real_pad / 16; ++i)
                            {
                                load2d(inputA.get_ptr(0) + i * k_real_pad * 16, L1A.get_ptr(0), i, k_real_pad / 16, m_real_pad / 16, false);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {

                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);                            
                            hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real_pad, m_real_pad, k_real, m_real, lda);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, m_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, k_real_pad, m_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < m_real_pad / 16; ++i)
                            {
                                load2d(inputA.get_ptr(0) + i * k_real_pad * 16, L1A.get_ptr(0), i, k_real_pad / 16, m_real_pad / 16, false);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }
                        
                        B_ptr = matrixB + col * n * ldb + j * m;

                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);
                        
                        hablas_load_matrix_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, ldb);

                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);
                        hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, alpha);

                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);
                        hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);

                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);

                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1);

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == 0) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);
                    }    
                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);

                }
            } else if(uplo == HABLAS_FILL_MODE_LOWER) {  //uplo == HABLAS_FILL_MODE_LOWER
                if (transA == HABLAS_OP_N) {
                    for (int j = 0; j < row + 1; j++) 
                    {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                        }
                        __gm__ half *A_ptr;
                        __gm__ half *B_ptr;
                        A_ptr = matrixA + j * m * lda + row * m;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == row) {
                            
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_diag_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, lda, diag, uplo);
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, (half)1.0);
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);

                        } else {
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, lda);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }
                        
                        B_ptr = matrixB + col * n * ldb + j * m;

                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);


                        hablas_load_matrix_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, ldb);
                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);

                        hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, alpha);
                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);

                        hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);
                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);

                        load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);

                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1);   

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == 0) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);
                    }

                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);
                } 
                else {    //transA == HABLAS_OP_T
                    for (int j = row; j < k_loop; j++) 
                    {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                        }
                        __gm__ half *A_ptr;
                        __gm__ half *B_ptr;
                        A_ptr = matrixA + j * m + row * m * lda;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == row) {

                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);
                            hablas_load_matrix_diag_gm2ub(ubA1, A_ptr, k_real_pad, m_real_pad, k_real, m_real, lda, diag, uplo);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, m_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, k_real_pad, m_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < m_real_pad / 16; ++i)
                            {
                                load2d(inputA.get_ptr(0) + i * k_real_pad * 16, L1A.get_ptr(0), i, k_real_pad / 16, m_real_pad / 16, false);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {

                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);                            
                            hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real_pad, m_real_pad, k_real, m_real, lda);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubA2, ubA1, k_real_pad, m_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, k_real_pad, m_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < m_real_pad / 16; ++i)
                            {
                                load2d(inputA.get_ptr(0) + i * k_real_pad * 16, L1A.get_ptr(0), i, k_real_pad / 16, m_real_pad / 16, false);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }
                        
                        B_ptr = matrixB + col * n * ldb + j * m;

                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);
                        
                        hablas_load_matrix_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, ldb);

                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);
                        hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, alpha);

                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);
                        hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);

                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);

                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1);

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == row) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);
                    }
 
                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);
                }
            }
            set_flag(PIPE_M, PIPE_V, 0);
            wait_flag(PIPE_M, PIPE_V, 0);
            hablas_store_matrixC_l02ub2ub(ub_buffer1, ub_buffer0, result.get_ptr(0), m_real_pad, n_real_pad);


            set_flag(PIPE_V, PIPE_MTE3, 3);
            wait_flag(PIPE_V, PIPE_MTE3, 3);
            hablas_store_matrixC_ub2gm(C_ptr, ub_buffer1, ub_buffer0, m_real_pad, n_real_pad, m_real, n_real, ldc);
            set_flag(PIPE_MTE3, PIPE_MTE2, 1);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, 1);
    } 
    else if (side == HABLAS_SIDE_RIGHT) {  
        int64_t m_tiles  = (M + n - 1) / n;
        int64_t n_tiles  = (N + m - 1) / m;
        int64_t k_loop = (N + m - 1) / m;

        int64_t m_remain = M % n;
        int64_t n_remain = N % m;
        int64_t k_remain = N % m;

        int64_t tiles_num = m_tiles * n_tiles;
        int64_t tiles_per_core = tiles_num / block_num;
        if (block_idx < tiles_num % block_num) {
             ++tiles_per_core;
        }

        set_flag(PIPE_MTE3, PIPE_MTE2, 1);
        for (int64_t i = 0; i < tiles_per_core; ++i) {
            int64_t id = i * block_num + block_idx;
            int64_t col = id / m_tiles;
            int64_t row = id % m_tiles;
            int64_t m_real = n;
            int64_t n_real = m;
            int64_t m_real_pad = n;
            int64_t n_real_pad = m;

            if (row == m_tiles - 1 && m_remain)
            {
                m_real = m_remain;
                m_real_pad = m_real % 16 ? (m_real & 0xFFFFFFF0) + 16 : m_real;
            }

            if (col == n_tiles - 1 && n_remain)
            {
                n_real = n_remain;
                n_real_pad = n_real % 16 ? (n_real & 0xFFFFFFF0) + 16 : n_real;
            }
            
            __gm__ half *C_ptr = matrixC + col * ldc * m +  row * n;
            wait_flag(PIPE_MTE3, PIPE_MTE2, 1);

            set_flag(PIPE_M, PIPE_MTE1, 0);
            set_flag(PIPE_V, PIPE_MTE2, 0);
            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
            set_flag(PIPE_MTE3, PIPE_V, 0);
            set_flag(PIPE_V, PIPE_MTE2, 1);
            set_flag(PIPE_MTE1, PIPE_MTE3, 1);
            set_flag(PIPE_MTE3, PIPE_V, 1);

            set_flag(PIPE_MTE3, PIPE_MTE2, 0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, 0);

            if(uplo == HABLAS_FILL_MODE_UPPER) { //上三角矩阵
                if(transA == HABLAS_OP_N) {
                    for(int j = 0; j < col + 1; j++) {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                        }
                        __gm__ half *A_ptr; //L0A位置上的矩阵，这里为matrixB
                        __gm__ half *B_ptr; //L0B位置上的矩阵，这里为matrixA

                        B_ptr = matrixA + j * m + col * m * lda;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == col) {

                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_diag_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, lda, diag, uplo);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);

                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {

                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, lda);
                            
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, (half)1.0);
                            
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);
                            
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        
                        }
                        
                        A_ptr = matrixB + j * m * ldb + row * n;
                        
                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);

                        hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, ldb);

                        
                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);
                        hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, alpha);

                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);
                        hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);
                        
                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);

                        load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);

                       
                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1); 

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == 0) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real_pad, k_real, n_real_pad, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real_pad, k_real, n_real_pad, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);
                    }

                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);
                }
                else {    //transA == HABLAS_OP_T
                    for(int j = col; j < k_loop; j++) {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                            // pipe_barrier(PIPE_ALL);
                        }
                        __gm__ half *A_ptr; //L0A位置上的矩阵，这里为matrixB
                        __gm__ half *B_ptr; //L0B位置上的矩阵，这里为matrixA

                        B_ptr = matrixA + j * m * lda + col * m;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == col) {
                        
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);
                            hablas_load_matrix_diag_gm2ub(ubB1, B_ptr, n_real_pad, k_real_pad, n_real, k_real, lda, diag, uplo);
                      
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, n_real_pad, k_real_pad, (half)1.0);
                       
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, n_real_pad, k_real_pad);
                        
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < k_real_pad / 16; ++i)
                            {
                                load2d(inputB.get_ptr(0) + i * n_real_pad * 16, L1B.get_ptr(0), i, n_real_pad / 16, k_real_pad / 16, true);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {
                 
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);
                            hablas_load_matrix_gm2ub(ubB1, B_ptr, n_real_pad, k_real_pad, n_real, k_real, lda);
                
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, n_real_pad, k_real_pad, (half)1.0);
                      
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, n_real_pad, k_real_pad);
                   
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < k_real_pad / 16; ++i)
                            {
                                load2d(inputB.get_ptr(0) + i * n_real_pad * 16, L1B.get_ptr(0), i, n_real_pad / 16, k_real_pad / 16, true);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }
                        
                        A_ptr = matrixB + j * m * ldb + row * n;
                   
                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);
                        hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, ldb);

                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);
                        hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, alpha);

                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);
                        hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);

                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);
                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1);

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == col) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);
                    } 
                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);
                }
            }
            else if(uplo == HABLAS_FILL_MODE_LOWER) {    //下三角矩阵
                if(transA == HABLAS_OP_N) {
                    for(int j = col; j < k_loop; j++) {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                        }
                        __gm__ half *A_ptr; //L0A位置上的矩阵，这里为matrixB
                        __gm__ half *B_ptr; //L0B位置上的矩阵，这里为matrixA

                        B_ptr = matrixA + j * m + col * m * lda;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == col) {
                         
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_diag_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, lda, diag, uplo);

                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, (half)1.0);

                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);
 
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);

                            hablas_load_matrix_gm2ub(ubB1, B_ptr, k_real_pad, n_real_pad, k_real, n_real, lda);
                        
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, k_real_pad, n_real_pad, (half)1.0);
                     
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, k_real_pad, n_real_pad);
                      
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            load2d(inputB.get_ptr(0), L1B.get_ptr(0), 0, (k_real_pad / 16) * (n_real_pad / 16), 1, false);
                            
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }
                        
                        A_ptr = matrixB + j * m * ldb + row * n;
                   
                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);

                        hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, ldb);

                 
                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);
                        hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, alpha);

                   
                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);
                        hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);
                    
                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);

                        load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);

               
                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1); 

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == col) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);        
                    }
                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);
                }
                else {    //transA == HABLAS_OP_T
                    for(int j = 0; j < col + 1; j++) {
                        int64_t k_real = m;
                        int64_t k_real_pad = m;
                        if (j == k_loop - 1 && k_remain > 0) {
                            k_real = k_remain;
                            k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
                            // pipe_barrier(PIPE_ALL);
                        }
                        __gm__ half *A_ptr; //L0A位置上的矩阵，这里为matrixB
                        __gm__ half *B_ptr; //L0B位置上的矩阵，这里为matrixA

                        B_ptr = matrixA + j * m * lda + col * m;
                        wait_flag(PIPE_M, PIPE_MTE1, 0);
                        if (j == col) {
                       
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);
                            hablas_load_matrix_diag_gm2ub(ubB1, B_ptr, n_real_pad, k_real_pad, n_real, k_real, lda, diag, uplo);
                     
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, n_real_pad, k_real_pad, (half)1.0);
                       
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, n_real_pad, k_real_pad);
                       
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < k_real_pad / 16; ++i)
                            {
                                load2d(inputB.get_ptr(0) + i * n_real_pad * 16, L1B.get_ptr(0), i, n_real_pad / 16, k_real_pad / 16, true);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        } else {
                       
                            wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                            wait_flag(PIPE_MTE3, PIPE_V, 0);
                            wait_flag(PIPE_V, PIPE_MTE2, 0);
                            hablas_load_matrix_gm2ub(ubB1, B_ptr, n_real_pad, k_real_pad, n_real, k_real, lda);
                   
                            set_flag(PIPE_MTE2, PIPE_V, 0);
                            wait_flag(PIPE_MTE2, PIPE_V, 0);
                            hablas_load_input_matrix_ND2zZ(ubB2, ubB1, n_real_pad, k_real_pad, (half)1.0);
                      
                            set_flag(PIPE_V, PIPE_MTE3, 0);
                            wait_flag(PIPE_V, PIPE_MTE3, 0);
                            hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, n_real_pad, k_real_pad);
                       
                            set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                            for (int i = 0; i < k_real_pad / 16; ++i)
                            {
                                load2d(inputB.get_ptr(0) + i * n_real_pad * 16, L1B.get_ptr(0), i, n_real_pad / 16, k_real_pad / 16, true);
                            }
                            set_flag(PIPE_V, PIPE_MTE2, 0);
                            set_flag(PIPE_MTE3, PIPE_V, 0);
                            set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                        }
                        
                        A_ptr = matrixB + j * m * ldb + row * n;
                   
                        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                        wait_flag(PIPE_MTE3, PIPE_V, 1);
                        wait_flag(PIPE_V, PIPE_MTE2, 1);
                        hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real_pad, k_real_pad, m_real, k_real, ldb);

                    
                        set_flag(PIPE_MTE2, PIPE_V, 1);
                        wait_flag(PIPE_MTE2, PIPE_V, 1);
                        hablas_load_input_matrix_ND2zZ(ubA2, ubA1, m_real_pad, k_real_pad, alpha);

                    
                        set_flag(PIPE_V, PIPE_MTE3, 1);
                        wait_flag(PIPE_V, PIPE_MTE3, 1);
                        hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);

                        set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                        load2d(inputA.get_ptr(0), L1A.get_ptr(0), 0, (m_real_pad / 16) * (k_real_pad / 16), 1, true);
                        set_flag(PIPE_V, PIPE_MTE2, 1);
                        set_flag(PIPE_MTE3, PIPE_V, 1);
                        set_flag(PIPE_MTE1, PIPE_MTE3, 1);

                        set_flag(PIPE_MTE1, PIPE_M, 0);
                        wait_flag(PIPE_MTE1, PIPE_M, 0);
                        if(j == 0) {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 1);
                        } else {
                            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real, k_real, n_real, 0);
                        }
                        set_flag(PIPE_M, PIPE_MTE1, 0);                
                    } 
                    wait_flag(PIPE_M, PIPE_MTE1, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 0);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                    wait_flag(PIPE_MTE3, PIPE_V, 0);
                    wait_flag(PIPE_V, PIPE_MTE2, 1);
                    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                    wait_flag(PIPE_MTE3, PIPE_V, 1);
                }
            }
            set_flag(PIPE_M, PIPE_V, 0);
            wait_flag(PIPE_M, PIPE_V, 0);
            hablas_store_matrixC_l02ub2ub(ub_buffer1, ub_buffer0, result.get_ptr(0), m_real_pad, n_real_pad);

            set_flag(PIPE_V, PIPE_MTE3, 3);
            wait_flag(PIPE_V, PIPE_MTE3, 3);
            hablas_store_matrixC_ub2gm(C_ptr, ub_buffer1, ub_buffer0, m_real_pad, n_real_pad, m_real, n_real, ldc);
            set_flag(PIPE_MTE3, PIPE_MTE2, 1);
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, 1);

    }

}
