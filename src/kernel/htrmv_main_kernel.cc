#include <hacl/hacl.h>

constexpr int UB_MAX_HALF_SIZE = 256 * 1024 / 2 - 32;
constexpr int L0AB_MAX_HALF_SIZE = 64 * 1024 / 2;
constexpr int L0C_MAX_SINGLE_SIZE = 256 * 1024 / 4;
constexpr int UB_MATRIX_SIZE = 128 * 128;
constexpr int UB_VECTOR_SIZE = 128;
constexpr int UB_WORKSPACE_SIZE = 128 * 128;

HACL_INLINE __aicore__ void hablas_memcpy(__gm__ half *dst, __ub__ half *src, int64_t len, int64_t space) {
    if (space < 16) {
        __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(dst, src, 1, 1, 0, 0);
    } else {
        _memcpy(dst, src, len);
    }
}

HACL_INLINE __aicore__ void __memcpy(__gm__ half *gm, __ub__ half *ub, int64_t len)
{
    int64_t nUnit = len / 16;
    int64_t unit_remain = len % 16;
    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(gm, ub, 1, nUnit, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, 0);
    wait_flag(PIPE_MTE3, PIPE_S, 0);
    if (unit_remain && len > 16)
    {
        int64_t offset_start = (len & 0xFFFFFFF0) - 16 + unit_remain;
        for (int i = 0; i < 16; ++i)
        {
            *(ub + i) = *(ub + offset_start + i);
        }
        set_flag(PIPE_S, PIPE_MTE3, 0);
        wait_flag(PIPE_S, PIPE_MTE3, 0);
        __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(gm + offset_start, ub, 1, 1, 0, 0);
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_gm2ub(__ub__ half *ub_ptr,
                                                     __gm__ half *gm_ptr, 
                                                     int64_t m_real, 
                                                     int64_t m_real_pad, 
                                                     int64_t n_real, 
                                                     int64_t n_real_pad, 
                                                     int64_t stride) 
{
    if (m_real % 16 || (stride - m_real) % 16) {
        for (int i = 0; i < n_real; ++i) {
pipe_barrier(PIPE_ALL);
            _memcpy(ub_ptr + i * m_real_pad, gm_ptr + i * stride, 1, m_real_pad / 16, 0, 0);
        }
    }
    else {
        _memcpy(ub_ptr, gm_ptr, n_real, m_real / 16, 0, (stride - m_real) / 16);
    }
}

HACL_INLINE __aicore__ void hablas_load_vector_gm2ub(__ub__ half *dst, 
                                                     __gm__ half *src, 
                                                     __ub__ half *wksp,
                                                     int64_t valid_len, 
                                                     int64_t inc) 
{
    if (inc == 1) {
        _memcpy(dst, src, valid_len);
    } else {
        int64_t content = UB_WORKSPACE_SIZE;
        int64_t loop = valid_len * inc / content;
        int64_t remain = valid_len * inc % content;
        int64_t start_posi = 0;
        int64_t iub = 0;
        for (int i = 0; i < loop; ++i) {
            pipe_barrier(PIPE_ALL);
            _memcpy(wksp, 
                    src + i * content,
                    content);
            pipe_barrier(PIPE_ALL);
            int iwhile = start_posi;
            while (iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                iwhile = iwhile + inc;
                iub = iub + 1;
            }
            pipe_barrier(PIPE_ALL);
            start_posi = iwhile - content;
        }
        if (remain) {
            pipe_barrier(PIPE_ALL);
            _memcpy(wksp, 
                    src + loop * content,
                    remain);
            pipe_barrier(PIPE_ALL);
            int iwhile = start_posi;
            while (iub < valid_len && iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                iwhile = iwhile + inc;
                iub = iub + 1;
            }
            pipe_barrier(PIPE_ALL);
        }
    } 
}

HACL_INLINE __aicore__ void hablas_load_matrix_ND2nZ(__ub__ half *dst,
                                                     __ub__ half *src, 
                                                     int64_t m_real_pad,
                                                     int64_t n_real_pad) 
{
    for (int64_t nd2zn_idx = 0; nd2zn_idx < m_real_pad / 16; ++nd2zn_idx) {
        __hacl_details__::__hacl_intrinsic_move_mask(128);
        __hacl_details__::__hacl_intrinsic_vec_adds(
            dst + nd2zn_idx * n_real_pad * 16,          // dst
            src + nd2zn_idx * 16,                       // src
            (half)0.0,                                  // half a
            n_real_pad * 16 / 128,                      // repeat_time
            8 * m_real_pad / 16,                        // src_stride
            8,                                          // dst_stride
            m_real_pad / 16,                            // src_block_stride
            1);                                         // dst_block_stride
    }
}

HACL_INLINE __aicore__ void hablas_load_matrix_ub2l1(__l1__ half *dst,
                                                     __ub__ half *src,
                                                     int64_t m_real_pad,
                                                     int64_t n_real_pad)
{
    _memcpy(dst, src, 1, m_real_pad * n_real_pad / 16, 0, 0);
}

HACL_INLINE __aicore__ void hablas_store_vector_l0c2ub(__ub__ half *dst,
                                                     __l0c__ float *src,
                                                     int64_t m_real)
{
    _memcpy(dst, 
            src,              
            (m_real + 15) / 16, // nburst 
            1,                  // burst_len 一次传输 1*16 的vector分形 
            0,                  // dst_gap 32B为单位 
            0,                  // src_gap 1024B为单位 
            VECTOR);            // 设为向量模式
}

HACL_INLINE __aicore__ void hablas_fill_zero(__ub__ half *dst,
                                             int64_t uplo,
                                             int64_t diag,
                                             int64_t m_real,
                                             int64_t m_real_pad,
                                             __ub__ half *uplo_matrix)
{
    __hacl_details__::__hacl_intrinsic_move_mask(m_real);
    __hacl_details__::__hacl_intrinsic_vec_mul<half>(
        dst,
        dst,
        uplo_matrix,
        m_real,// repeat times
        m_real_pad / 16, // dst repeat stride
        m_real_pad / 16, // src0 repeat stride
        128 / 16, // src1 repeat stride
        1,// dst block stride
        1,// src0 block stride
        1// src1 block stride
    );
    if (diag) {
        set_flag(PIPE_V, PIPE_S, 3);
        wait_flag(PIPE_V, PIPE_S, 3);
        for (int64_t i = 0; i < m_real; ++i) {
            for (int j = i; j <= i; ++j) {
                *(dst + i * m_real_pad + j) = 1.0;
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, 3);
    wait_flag(PIPE_S, PIPE_V, 3);
}


extern "C" __global__ __aicore__  
void hablas_htrmv_kernel(int64_t uplo,
                        int64_t transA,
                        int64_t diag,
                        int64_t M,
                        __gm__ half *matrixA,
                        int64_t lda,
                        __gm__ half *vectorX,
                        int64_t incx,
                        __gm__ half *workspace,
                        int64_t base_block_size)
// extern "C" __global__ __aicore__  
// void hablas_htrmv_kernel(
//                     __gm__ half *matrixA,
//                     __gm__ half *vectorX,
//                     __gm__ half *workspace)
{
    // int64_t uplo = 1;
    // int64_t transA = 0;
    // int64_t diag = 0;
    // int64_t M = 1024;
    // int64_t lda = 1024;
    // int64_t incx = 1;
    Vector<float_8, L0C_MAX_SINGLE_SIZE / 8, HACL_L0C> result;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0A> inputA;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L0B> inputB;

    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1A;
    Vector<half_16, L0AB_MAX_HALF_SIZE / 16, HACL_L1> L1B;

    Vector<half_16, UB_MAX_HALF_SIZE / 16, HACL_UB> ub;

    __ub__ half *ubA1 = ub.get_ptr(0);
    __ub__ half *ubA2 = ub.get_ptr(0) + 2 * UB_MATRIX_SIZE;
    __ub__ half *ubX1 = ub.get_ptr(0) + 4 * UB_MATRIX_SIZE;
    __ub__ half *ubW1 = ub.get_ptr(0) + 5 * UB_MATRIX_SIZE + UB_VECTOR_SIZE;
    __ub__ half *ubR1 = ub.get_ptr(0) + 6 * UB_MATRIX_SIZE + 2 * UB_VECTOR_SIZE;
    __ub__ half *ub_uplo_matrix = ub.get_ptr(0) + 6 * UB_MATRIX_SIZE + 4 * UB_VECTOR_SIZE;


    if (uplo) {
        __hacl_details__::__hacl_intrinsic_move_mask(128);
        __hacl_details__::__hacl_intrinsic_vec_dup(
            ub_uplo_matrix, //dst
            half(0.0), //src
            128, // repeat times
            128 / 16, // dst repeat stride
            1 // dst block stride
        );
        set_flag(PIPE_V, PIPE_S, 2);
        wait_flag(PIPE_V, PIPE_S, 2);
        for (int i = 0; i < 128; ++i) {
            vec_dup(ub_uplo_matrix + 128 * i, half(1.0), i + 1);
        }
    } else {
        __hacl_details__::__hacl_intrinsic_move_mask(128);
        __hacl_details__::__hacl_intrinsic_vec_dup(
            ub_uplo_matrix, //dst
            half(1.0), //src
            128, // repeat times
            128 / 16, // dst repeat stride
            1 // dst block stride
        );
        set_flag(PIPE_V, PIPE_S, 2);
        wait_flag(PIPE_V, PIPE_S, 2);
        for (int i = 0; i < 128; ++i) {
            set_flag(PIPE_S, PIPE_V, 2);
            wait_flag(PIPE_S, PIPE_V, 2);
            vec_dup(ub_uplo_matrix + 128 * i, half(0.0), i + 1);
            set_flag(PIPE_V, PIPE_S, 2);
            wait_flag(PIPE_V, PIPE_S, 2);
            *(ub_uplo_matrix + 128 * i + i) = 1.0;
        }
    }
pipe_barrier(PIPE_ALL);

    int64_t m = base_block_size;

    int64_t m_tiles  = (M + m - 1) / m;
    int64_t n_tiles  = 1;
    int64_t k_loop   = (M + m - 1) / m;
    int64_t m_remain = M % m;
    int64_t k_remain = M % m;

    int64_t tiles_num = m_tiles * n_tiles;
    int64_t tiles_per_core = tiles_num / block_num;
    if (block_idx < tiles_num % block_num) {
        ++tiles_per_core;
    }

    set_flag(PIPE_V, PIPE_MTE2, 0);
    set_flag(PIPE_MTE3, PIPE_V, 0);
    set_flag(PIPE_MTE1, PIPE_MTE3, 0);
    set_flag(PIPE_M, PIPE_MTE1, 0);
    set_flag(PIPE_MTE3, PIPE_MTE2, 1);
    set_flag(PIPE_MTE1, PIPE_MTE3, 1);
    set_flag(PIPE_M, PIPE_MTE1, 1);
    set_flag(PIPE_V, PIPE_M, 0);
    set_flag(PIPE_MTE3, PIPE_V, 2);

    for (int i = 0; i < tiles_per_core; i++) {
        int64_t block_index = i * block_num + block_idx;
        int64_t row = block_index / n_tiles;
        int64_t m_real = m;
        if (row == m_tiles - 1 && m_remain > 0) {
            m_real = m_remain;
        }
        
        int64_t m_real_pad = m_real % 16 ? (m_real & 0xFFFFFFFFFFFFFFF0) + 16 : m_real;

        __gm__ half *W_ptr = workspace + row * m;

        // uplo = 1上三角矩阵 
        // 矩阵A在右边 向量X在左边 启用GEMV模式
        int64_t k_idx = row;
        int64_t k_dst = k_loop;
        if (transA - uplo == 0) {
            k_idx = 0;
            k_dst = row + 1;
        }
        // 转置并且下三角等于上三角
        // if (!transA && !uplo) {
        //     k_idx = 0;
        //     k_dst = row + 1;
        // }

        for (; k_idx < k_dst; ++k_idx) {  
            int64_t k_real = m;
            if (k_idx == k_loop - 1 && k_remain > 0) {
                k_real = k_remain;
            }
            int64_t k_real_pad = k_real % 16 ? (k_real & 0xFFFFFFFFFFFFFFF0) + 16 : k_real;

            __gm__ half *X_ptr = vectorX + m * incx * k_idx;

            if (transA == 1) {
                __gm__ half *A_ptr = matrixA + m * lda * row + k_idx * m;

                wait_flag(PIPE_V, PIPE_MTE2, 0); // 等待ubA1缓冲区读完
                hablas_load_matrix_gm2ub(ubA1, A_ptr, k_real, k_real_pad, m_real, m_real_pad, lda);

                set_flag(PIPE_MTE2, PIPE_V, 0);
                wait_flag(PIPE_MTE2, PIPE_V, 0);
                if (k_idx == row) {
                    set_flag(PIPE_MTE2, PIPE_S, 0);
                    wait_flag(PIPE_MTE2, PIPE_S, 0); 
                    hablas_fill_zero(ubA1, uplo, diag, m_real, m_real_pad, ub_uplo_matrix);
                }

                wait_flag(PIPE_MTE3, PIPE_V, 0); //等待ubA2缓冲区读完
                hablas_load_matrix_ND2nZ(ubA2, ubA1, k_real_pad, m_real_pad);
                set_flag(PIPE_V, PIPE_MTE2, 0); // ubA1缓冲区读完

                set_flag(PIPE_V, PIPE_MTE3, 0);
                wait_flag(PIPE_V, PIPE_MTE3, 0);

                wait_flag(PIPE_MTE1, PIPE_MTE3, 0); // 等待L1A读完
                hablas_load_matrix_ub2l1(L1A.get_ptr(0), ubA2, m_real_pad, k_real_pad);
                set_flag(PIPE_MTE3, PIPE_V, 0); // ubA2缓冲区读完

                set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE3, PIPE_MTE1, 0);
                wait_flag(PIPE_M, PIPE_MTE1, 0); // 等待inputB读完
                load2d(inputB.get_ptr(0),
                    L1A.get_ptr(0),
                    0,                                     // index
                    (m_real_pad / 16) * (k_real_pad / 16), // repeat
                    1,                                     // src_stride
                    0);                                    // transpose
                set_flag(PIPE_MTE1, PIPE_MTE3, 0); // L1A读完
                set_flag(PIPE_MTE1, PIPE_M, 0);
            } else {
                __gm__ half *A_ptr = matrixA + m * row + k_idx * m * lda;
                wait_flag(PIPE_V, PIPE_MTE2, 0); // 等待ubA1缓冲区读完

                hablas_load_matrix_gm2ub(ubA1, A_ptr, m_real, m_real_pad, k_real, k_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_V, 0);
                wait_flag(PIPE_MTE2, PIPE_V, 0);
                if (k_idx == row) {
                    set_flag(PIPE_MTE2, PIPE_S, 0);
                    wait_flag(PIPE_MTE2, PIPE_S, 0); 
                    hablas_fill_zero(ubA1, uplo, diag, m_real, m_real_pad, ub_uplo_matrix);
                }

                wait_flag(PIPE_MTE3, PIPE_V, 0); //等待ubA2缓冲区读完
                hablas_load_matrix_ND2nZ(ubA2, ubA1, m_real_pad, k_real_pad);
                set_flag(PIPE_V, PIPE_MTE2, 0); // ubA1缓冲区读完

                set_flag(PIPE_V, PIPE_MTE3, 0);
                wait_flag(PIPE_V, PIPE_MTE3, 0);

                wait_flag(PIPE_MTE1, PIPE_MTE3, 0); // 等待L1A读完
                hablas_load_matrix_ub2l1(L1A.get_ptr(0), ubA2, k_real_pad, m_real_pad);

                set_flag(PIPE_MTE3, PIPE_V, 0); // ubA2缓冲区读完

                set_flag(PIPE_MTE3, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE3, PIPE_MTE1, 0);

                wait_flag(PIPE_M, PIPE_MTE1, 0); // 等待inputB读完
               
                for (int i = 0; i < k_real_pad / 16; ++i) {
                    load2d(inputB.get_ptr(0) + i * m_real_pad * 16, 
                           L1A.get_ptr(0), 
                           i,                   // index 
                           m_real_pad / 16,     // repeat 
                           k_real_pad / 16,     // src_stride 
                           1);                  // transpose
                }
                set_flag(PIPE_MTE1, PIPE_MTE3, 0); // L1A读完
                set_flag(PIPE_MTE1, PIPE_M, 0);
            }

            wait_flag(PIPE_MTE3, PIPE_MTE2, 1);//等待ubX1读完
            hablas_load_vector_gm2ub(ubX1, X_ptr, ubW1, k_real, incx);

            set_flag(PIPE_MTE2, PIPE_MTE3, 1);
            wait_flag(PIPE_MTE2, PIPE_MTE3, 1);
            wait_flag(PIPE_MTE1, PIPE_MTE3, 1); // 等待L1B读完
            hablas_load_matrix_ub2l1(L1B.get_ptr(0), ubX1, 1, k_real_pad);

            set_flag(PIPE_MTE3, PIPE_MTE2, 1);// ubX1读完
            set_flag(PIPE_MTE3, PIPE_MTE1, 1);
            wait_flag(PIPE_MTE3, PIPE_MTE1, 1);

            wait_flag(PIPE_M, PIPE_MTE1, 1);// 等待inputA读完
            load2d(inputA.get_ptr(0),
                    L1B.get_ptr(0),
                    0,                                     // index
                    ((k_real_pad + 255) / 256),            // repeat
                    1,                                     // src_stride
                    0);                                    // transpose 
            set_flag(PIPE_MTE1, PIPE_MTE3, 1); // L1B读完
            set_flag(PIPE_MTE1, PIPE_M, 1);

     
            int64_t init = (k_idx == 0) || (k_idx == row && (transA - uplo)) ? 1 : 0; 

            wait_flag(PIPE_MTE1, PIPE_M, 0);
            wait_flag(PIPE_MTE1, PIPE_M, 1);
            if (k_idx == 0 || (((transA - uplo) != 0) && k_idx == row)) {
                wait_flag(PIPE_V, PIPE_M, 0);// 等待result读完 
            }
            mmad(result.get_ptr(0), inputA.get_ptr(0), inputB.get_ptr(0), 1, k_real, m_real, init);
            set_flag(PIPE_M, PIPE_MTE1, 1); // inputA读完
            set_flag(PIPE_M, PIPE_MTE1, 0); // inputB读完
            if (k_idx == k_dst - 1) {
                set_flag(PIPE_M, PIPE_V, 0);
            } 
        }
        wait_flag(PIPE_MTE3, PIPE_V, 2);// 等待ubR1读完 
        wait_flag(PIPE_M, PIPE_V, 0);
        hablas_store_vector_l0c2ub(ubR1, result.get_ptr(0), m_real);
        set_flag(PIPE_V, PIPE_M, 0);// result读完
 
        set_flag(PIPE_V, PIPE_MTE3, 2);
        wait_flag(PIPE_V, PIPE_MTE3, 2);
        set_flag(PIPE_V, PIPE_S, 2);
        wait_flag(PIPE_V, PIPE_S, 2);

        hablas_memcpy(W_ptr, ubR1, m_real, M);
        set_flag(PIPE_MTE3, PIPE_V, 2);// ubR1读完 
    }

    wait_flag(PIPE_V, PIPE_MTE2, 0);
    wait_flag(PIPE_MTE3, PIPE_V, 0);
    wait_flag(PIPE_MTE1, PIPE_MTE3, 0);
    wait_flag(PIPE_M, PIPE_MTE1, 0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, 1);
    wait_flag(PIPE_MTE1, PIPE_MTE3, 1);
    wait_flag(PIPE_M, PIPE_MTE1, 1);
    wait_flag(PIPE_V, PIPE_M, 0);
    wait_flag(PIPE_MTE3, PIPE_V, 2);
}