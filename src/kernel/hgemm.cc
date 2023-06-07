#include "tools.h"

#ifndef CAMODEL_PROFILING
extern "C" __global__ __aicore__ void hablas_hgemm_kernel(hablasOperation_t transA,
                                                          hablasOperation_t transB,
                                                          int64_t M,
                                                          int64_t N,
                                                          int64_t K,
                                                          half alpha,
                                                          __gm__ half *matrixA,
                                                          int64_t lda,
                                                          __gm__ half *matrixB,
                                                          int64_t ldb,
                                                          half beta,
                                                          __gm__ half *matrixC,
                                                          int64_t ldc)
#else
extern "C" __global__ __aicore__ void hablas_hgemm_kernel(__gm__ half *matrixA,
                                                          __gm__ half *matrixB,
                                                          __gm__ half *matrixC)
#endif
{
    Vector<float_8, L0C_MAX_SINGLE_SIZE / 8, HACL_L0C> result;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0A> inputA;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0B> inputB;

    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1A;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1B;
    Vector<half_16, L0C_MAX_HALF_SIZE / 16, HACL_L1> L1C;

    Vector<half_16, UB_MAX_HALF_SIZE / 16, HACL_UB> ub;

#ifdef CAMODEL_PROFILING
    hablasOperation_t transA = HABLAS_OP_N;
    hablasOperation_t transB = HABLAS_OP_N;
    int64_t M = 2880;
    int64_t K = 1071;
    int64_t N = 1122;
    int64_t lda = M;
    int64_t ldb = K;
    int64_t ldc = M;
    half alpha = 0.5;
    half beta = 0.5;
#endif

    int64_t m = 256;
    int64_t n = 256;
    int64_t k = 128;

    int64_t m_remain = M % m;
    int64_t n_remain = N % n;
    int64_t k_remain = K % k;
     
    while (M >= 16 && m_remain && m_remain < 16)
    {
        m -= 16;
        m_remain = M % m;
    }
    int64_t m_tiles = (M + m - 1) / m;
    int64_t n_tiles = (N + n - 1) / n;
    int64_t k_loop = (K + k - 1) / k;

    int64_t tiles_num = m_tiles * n_tiles;
    int64_t tiles_per_core = tiles_num / block_num;
    if (block_idx < tiles_num % block_num)
        ++tiles_per_core;

    __ub__ half *ubA1 = ub.get_ptr(0);
    __ub__ half *ubA2 = ubA1 + UB_HALF_64KB;
    __ub__ half *ubB1 = ubA2 + UB_HALF_64KB;
    __ub__ half *ubB2 = ubB1 + UB_HALF_64KB;

    __ub__ half *ub_buffer0 = ub.get_ptr(0);
    __ub__ half *ub_buffer1 = ub_buffer0 + 2 * UB_HALF_64KB;

    set_flag(PIPE_MTE3, PIPE_MTE2, 1);
    set_flag(PIPE_MTE1, PIPE_MTE2, 0);
    for (int64_t i = 0; i < tiles_per_core; ++i)
    {
        int64_t id = i * block_num + block_idx;
        int64_t col = id / m_tiles;
        int64_t row = id % m_tiles;
        int64_t m_real = m;
        int64_t n_real = n;
        int64_t m_real_pad = m;
        int64_t n_real_pad = n;
        {
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
        }
        // prefetch C to ub_c
        __gm__ half *C_ptr = matrixC + col * n * ldc + row * m;
         
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
         
        for (int64_t j = 0; j < k_loop; ++j)
        {
            int64_t k_real = k;
            int64_t k_real_pad = k;
            if (j == k_loop - 1 && k_remain)
            {
                k_real = k_remain;
                k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFF0) + 16 : k_real;
            }
            __gm__ half *A_ptr = matrixA + j * k * lda + row * m;
            __gm__ half *B_ptr = matrixB + col * n * ldb + j * k;

            int64_t A_m = m_real_pad, A_n = k_real_pad, A_m_real = m_real, A_n_real = k_real;
            int64_t B_m = k_real_pad, B_n = n_real_pad, B_m_real = k_real, B_n_real = n_real;

            if (transA == HABLAS_OP_T)
            {
                A_ptr = matrixA + row * m * lda + j * k;
                int64_t tmp = A_m;
                A_m = A_n;
                A_n = tmp;
                tmp = A_m_real;
                A_m_real = A_n_real;
                A_n_real = tmp;
            }

            if (transB == HABLAS_OP_T)
            {
                B_ptr = matrixB + j * k * ldb + col * n;
                int64_t tmp = B_m;
                B_m = B_n;
                B_n = tmp;
                tmp = B_m_real;
                B_m_real = B_n_real;
                B_n_real = tmp;
            }
             
            wait_flag(PIPE_M, PIPE_MTE1, 0);
            // A
            {
                wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
                wait_flag(PIPE_MTE3, PIPE_V, 0);
                wait_flag(PIPE_V, PIPE_MTE2, 0);

                hablas_load_matrix_gm2ub(ubA1, A_ptr, A_m, A_n, A_m_real, A_n_real, lda);
                 

                set_flag(PIPE_MTE2, PIPE_V, 0);
                wait_flag(PIPE_MTE2, PIPE_V, 0);
                hablas_load_input_matrix_ND2zZ(ubA2, ubA1, A_m, A_n, (half)1.0);
                 

                set_flag(PIPE_V, PIPE_MTE3, 0);
                wait_flag(PIPE_V, PIPE_MTE3, 0);
                hablas_load_input_matrix_ub2l1(L1A.get_ptr(0), ubA2, A_m, A_n);
                set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                 
                hablas_load_l12l0a(inputA.get_ptr(0), L1A.get_ptr(0), A_m, A_n, transA);

                set_flag(PIPE_V, PIPE_MTE2, 0);
                set_flag(PIPE_MTE3, PIPE_V, 0);
                set_flag(PIPE_MTE1, PIPE_MTE3, 0);
                 
            }
            // B
            {
                wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
                wait_flag(PIPE_MTE3, PIPE_V, 1);
                wait_flag(PIPE_V, PIPE_MTE2, 1);

                hablas_load_matrix_gm2ub(ubB1, B_ptr, B_m, B_n, B_m_real, B_n_real, ldb);
                set_flag(PIPE_MTE2, PIPE_V, 1);
                wait_flag(PIPE_MTE2, PIPE_V, 1);
                 
                hablas_load_input_matrix_ND2zZ(ubB2, ubB1, B_m, B_n, alpha);
                set_flag(PIPE_V, PIPE_MTE3, 1);
                wait_flag(PIPE_V, PIPE_MTE3, 1);
                 
                hablas_load_input_matrix_ub2l1(L1B.get_ptr(0), ubB2, B_m, B_n);
                set_flag(PIPE_MTE3, PIPE_MTE1, 1);
                wait_flag(PIPE_MTE3, PIPE_MTE1, 1);
                 
                hablas_load_l12l0b(inputB.get_ptr(0), L1B.get_ptr(0), B_m, B_n, transB);

                set_flag(PIPE_V, PIPE_MTE2, 1);
                set_flag(PIPE_MTE3, PIPE_V, 1);
                set_flag(PIPE_MTE1, PIPE_MTE3, 1);
                 
            }
             
            set_flag(PIPE_MTE1, PIPE_M, 0);
            wait_flag(PIPE_MTE1, PIPE_M, 0);
            if (j == 0)
            {
                mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real_pad, k_real, n_real_pad, 1);
            }
            else
            {
                mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), m_real_pad, k_real, n_real_pad, 0);
            }
            set_flag(PIPE_M, PIPE_MTE1, 0);
             
        }
#ifndef TBE
        wait_flag(PIPE_MTE1, PIPE_MTE2, 0);
        hablas_load_matrix_gm2l1(L1C.get_ptr(0), C_ptr, m_real_pad, n_real_pad, m_real, n_real, ldc);
         
        set_flag(PIPE_MTE2, PIPE_MTE1, 0);
#endif
        wait_flag(PIPE_M, PIPE_MTE1, 0);
        wait_flag(PIPE_V, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 0);
        wait_flag(PIPE_V, PIPE_MTE2, 1);
        wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
        wait_flag(PIPE_MTE3, PIPE_V, 1);

        set_flag(PIPE_M, PIPE_V, 0);
        wait_flag(PIPE_M, PIPE_V, 0);

        hablas_store_matrixC_l02ub2ub(ub_buffer1, ub_buffer0, result.get_ptr(0), m_real_pad, n_real_pad);
         
#ifndef TBE
        set_flag(PIPE_V, PIPE_MTE1, 0);
        wait_flag(PIPE_V, PIPE_MTE1, 0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, 0);
        hablas_load_matrix_l12ub(ub_buffer0, L1C.get_ptr(0), m_real_pad, n_real_pad);
        set_flag(PIPE_MTE1, PIPE_MTE2, 0);
        set_flag(PIPE_MTE1, PIPE_V, 0);
        wait_flag(PIPE_MTE1, PIPE_V, 0);
        vec_axpy(ub_buffer1, ub_buffer0, beta, m_real_pad * n_real);
#endif
        set_flag(PIPE_V, PIPE_MTE3, 3);
        wait_flag(PIPE_V, PIPE_MTE3, 3);
        hablas_store_matrixC_ub2gm(C_ptr, ub_buffer1, m_real_pad, n_real_pad, m_real, n_real, ldc);
        set_flag(PIPE_MTE3, PIPE_MTE2, 1);
         
    }
     
    wait_flag(PIPE_MTE3, PIPE_MTE2, 1);
    wait_flag(PIPE_MTE1, PIPE_MTE2, 0);
     
}