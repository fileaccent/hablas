#include <hacl/hacl.h>

constexpr int BLOCK_DIM = 80;
constexpr int UB_MATRIX_SIZE = BLOCK_DIM * BLOCK_DIM;
constexpr int UB_VECTOR_SIZE = BLOCK_DIM;
constexpr int UB_TMP_BLOCK_SIZE = BLOCK_DIM * 64 * 2; // 用于乘法运算存储中间结果
constexpr int UB_WORKSPACE_SIZE = BLOCK_DIM * BLOCK_DIM; // 用于搬运向量gm到ub inc不为1的情况 存储中间搬运结果 

HACL_INLINE __aicore__ void hablas_memcpy(__gm__ float *dst, __ub__ float *src, int64_t len, int64_t space) {
    if (space < 8) {
        __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(dst, src, 1, 1, 0, 0);
    } else {
        _memcpy(dst, src, len);
    }
}

HACL_INLINE __aicore__ void 
hablas_load_cmatrix_gm2ub(__ub__ float *dst,
                          __gm__ float *src,
                          int64_t m_real,
                          int64_t n_real,
                          int64_t m_real_pad,
                          int64_t n_real_pad,
                          int64_t stride) {
    if (m_real % 8 || (stride - m_real) % 8) {
        for (int j = 0; j < n_real; ++j) {
            _memcpy(dst + j * m_real_pad * 2, src + j * stride * 2, 1, m_real_pad / 8 * 2, 0, 0);
        }
    }
    else {
        _memcpy(dst, src, n_real, m_real / 8 * 2, 0, (stride - m_real) / 8 * 2);
    }
}

HACL_INLINE __aicore__ void
hablas_complex_to_real_imag(__ub__ float *dst,
                            __ub__ float *src,
                            __ub__ float *vector,
                            int64_t m_real_pad,
                            int64_t n_real_pad,
                            int64_t m_real,
                            int64_t n_real) 
{
     int64_t loop = m_real * 2 / 64;
    int64_t remain = m_real * 2 % 64;

    for (int loop_idx = 0; loop_idx < loop; ++loop_idx) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_mul<float>(
            dst + loop_idx * 64,
            src + loop_idx * 64,
            vector + loop_idx * 64,
            n_real,// repeat times
            m_real_pad * 2 / 8, // dst repeat stride
            m_real_pad * 2 / 8, // src0 repeat stride
            0, // src1 repeat stride
            1,// dst block stride
            1,// src0 block stride
            1// src1 block stride
        );
    }
    if (remain) {
        __hacl_details__::__hacl_intrinsic_move_mask(remain);
        __hacl_details__::__hacl_intrinsic_vec_mul<float>(
            dst + loop * 64,
            src + loop * 64,
            vector + loop * 64,
            n_real,// repeat times
            m_real_pad * 2 / 8, // dst repeat stride
            m_real_pad * 2 / 8, // src0 repeat stride
            0, // src1 repeat stride
            1,// dst block stride
            1,// src0 block stride
            1// src1 block stride
        );
    }
}

HACL_INLINE __aicore__ void 
hablas_load_cvector_gm2ub(__ub__ float *dst, 
                          __gm__ float *src, 
                          __ub__ float *wksp,
                          int64_t valid_len, 
                          int64_t inc) 
{
    if (inc == 1) {
        _memcpy(dst, src, valid_len * 2);
    } else {
        int32_t content = UB_WORKSPACE_SIZE;
        int32_t loop = valid_len * inc * 2 / content;
        int32_t remain = valid_len * inc * 2 % content;
        int32_t start_posi = 0;
        int32_t iub = 0;
        for (int i = 0; i < loop; ++i) {
            pipe_barrier(PIPE_ALL);
            _memcpy(wksp, 
                    src + i * content,
                    content);
            pipe_barrier(PIPE_ALL);
            int iwhile = start_posi;
            while (iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                *(dst + iub + 1) = *(wksp + iwhile + 1);
                iwhile = iwhile + inc * 2;
                iub = iub + 2;
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
            while (iub < valid_len * 2 && iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                *(dst + iub + 1) = *(wksp + iwhile + 1);
                iwhile = iwhile + inc * 2;
                iub = iub + 2;
            }
            pipe_barrier(PIPE_ALL);
        }
    }
}

HACL_INLINE __aicore__ void 
hablas_store_cvector_ub2gm(__gm__ float *dst, 
                           __ub__ float *src, 
                           int64_t valid_len) 
{
    _memcpy(dst, src, valid_len * 2);
}

HACL_INLINE __aicore__ void
hablas_matrix_vector_muls_notrans(__ub__ float *dst,
                                  __ub__ float *src0,
                                  __ub__ float *src1,
                                  int64_t m_real,
                                  int64_t n_real,
                                  int64_t m_real_pad,
                                  int64_t n_real_pad,
                                  int64_t flag)
{
    for (int64_t n_idx = 0; n_idx < n_real; ++n_idx) {
        float t = *(src1 + n_idx * 2);
        set_flag(PIPE_S, PIPE_V, 3);
        wait_flag(PIPE_S, PIPE_V, 3);
        if (flag) t = -t; 
        vec_axpy(dst, src0 + m_real_pad * n_idx, t, m_real);
    }
}

HACL_INLINE __aicore__ void
hablas_matrix_vector_muls_trans(__ub__ float *dst,
                                __ub__ float *src0,
                                __ub__ float *src1,
                                __ub__ float *tmp,
                                int64_t m_real,
                                int64_t n_real,
                                int64_t m_real_pad,
                                int64_t n_real_pad,
                                int64_t flag)
{
    vec_dup(tmp, (float)0, UB_TMP_BLOCK_SIZE); // tmp块内容清零
    int64_t loop = n_real * 2 / 64;
    int64_t remain = n_real * 2 % 64;
    for (int64_t idx = 0; idx < loop; ++idx) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_mla(tmp,
                                 src0 + 64 * idx,
                                 src1 + 64 * idx,
                                 m_real,//repeat times
                                 8 * 2,// dst repeat stride
                                 n_real_pad * 2 / 8, // src0 repeat stride
                                 0,// src1 repeat stride
                                 1,// dst block stride
                                 1,// src0 block stride
                                 1// src1 block stride
                                 );
    }
    if (remain) {
        __hacl_details__::__hacl_intrinsic_move_mask(remain);
        __hacl_details__::__hacl_intrinsic_vec_mla(tmp,
                                 src0 + 64 * loop,
                                 src1 + 64 * loop,
                                 m_real,//repeat times
                                 8 * 2,// dst repeat stride
                                 n_real_pad * 2 / 8, // src0 repeat stride
                                 0,// src1 repeat stride
                                 1,// dst block stride
                                 1,// src0 block stride
                                 1// src1 block stride
                                 );
    }
    __hacl_details__::__hacl_intrinsic_move_mask(64);
    __hacl_details__::__hacl_intrinsic_vec_reduce_add(tmp,
                                    tmp,
                                    m_real * 2, //repeat times
                                    8, //src repeat stide
                                    1  //src block stride
                                    );
    if (flag) {
        vec_sub(dst, dst, tmp, m_real * 2);
    } else {
        vec_add(dst, dst, tmp, m_real * 2);
    }
}

HACL_INLINE __aicore__ void
hablas_complex_muls_trans(__ub__ float *real_dst,
                            __ub__ float *imag_dst,
                            __ub__ float *real_src0,
                            __ub__ float *imag_src0,
                            __ub__ float *real_src1,
                            __ub__ float *imag_src1,
                            __ub__ float *tmp,
                            int64_t m_real,
                            int64_t n_real,
                            int64_t m_real_pad,
                            int64_t n_real_pad) 
{
    hablas_matrix_vector_muls_trans(real_dst, real_src0, real_src1, tmp, m_real, n_real, m_real_pad, n_real_pad, 0);
    hablas_matrix_vector_muls_trans(real_dst, imag_src0, imag_src1, tmp, m_real, n_real, m_real_pad, n_real_pad, 1);
    hablas_matrix_vector_muls_trans(imag_dst, real_src0, imag_src1, tmp, m_real, n_real, m_real_pad, n_real_pad, 0);
    hablas_matrix_vector_muls_trans(imag_dst, imag_src0, real_src1, tmp, m_real, n_real, m_real_pad, n_real_pad, 0);
}

HACL_INLINE __aicore__ void
hablas_complex_muls_notrans(__ub__ float *real_dst,
                            __ub__ float *imag_dst,
                            __ub__ float *real_src0,
                            __ub__ float *imag_src0,
                            __ub__ float *real_src1,
                            __ub__ float *imag_src1,
                            int64_t m_real,
                            int64_t n_real,
                            int64_t m_real_pad,
                            int64_t n_real_pad) 
{
    hablas_matrix_vector_muls_notrans(real_dst, real_src0, real_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
    hablas_matrix_vector_muls_notrans(real_dst, imag_src0, imag_src1, m_real, n_real, m_real_pad, n_real_pad, 1);
    hablas_matrix_vector_muls_notrans(imag_dst, real_src0, imag_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
    hablas_matrix_vector_muls_notrans(imag_dst, imag_src0, real_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
}

HACL_INLINE __aicore__ void hablas_fill_zero(__ub__ float *dst,
                                             int32_t uplo,
                                             int32_t diag,
                                             int32_t m_real,
                                             int32_t m_real_pad,
                                             int32_t isImag,
                                             __ub__ float *uplo_matrix)
{
    int64_t loop = m_real_pad * 2 / 64;
    int64_t remain = m_real_pad * 2 % 64;
    for (int64_t loop_idx = 0; loop_idx < loop; ++loop_idx) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_mul<float>(
            dst + 64 * loop_idx,
            dst + 64 * loop_idx,
            uplo_matrix + 64 * loop_idx,
            m_real,// repeat times
            m_real_pad * 2 / 8, // dst repeat stride
            m_real_pad * 2 / 8, // src0 repeat stride
            BLOCK_DIM * 2 / 8, // src1 repeat stride
            1,// dst block stride
            1,// src0 block stride
            1// src1 block stride
        );
    }
    if (remain) {
        __hacl_details__::__hacl_intrinsic_move_mask(remain);
        __hacl_details__::__hacl_intrinsic_vec_mul<float>(
            dst + 64 * loop,
            dst + 64 * loop,
            uplo_matrix + 64 * loop,
            m_real,// repeat times
            m_real_pad * 2 / 8, // dst repeat stride
            m_real_pad * 2 / 8, // src0 repeat stride
            BLOCK_DIM * 2 / 8, // src1 repeat stride
            1,// dst block stride
            1,// src0 block stride
            1// src1 block stride
        );
    }
    set_flag(PIPE_V, PIPE_S, 3);
    wait_flag(PIPE_V, PIPE_S, 3);
    if (diag) {
        for (int32_t i = 0; i < m_real; ++i) {
            for (int j = i; j <= i; ++j) {
                if (isImag) {
                    *(dst + i * m_real_pad * 2 + 2 * j) = 0.0;
                    *(dst + i * m_real_pad * 2 + 2 * j + 1) = 0.0;
                } else {
                    *(dst + i * m_real_pad * 2 + 2 * j) = 1.0;
                    *(dst + i * m_real_pad * 2 + 2 * j + 1) = 0.0;
                }
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, 3);
    wait_flag(PIPE_S, PIPE_V, 3);
}

HACL_INLINE __aicore__ void
hablas_generate_uplo_matrix(__ub__ float *uplo_matrix, int64_t uplo) {
    int64_t block_size = BLOCK_DIM;
    int64_t loop = block_size * 2 / 64;
    int64_t remain = block_size * 2 % 64;
    if (uplo) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        for (int idx = 0; idx < loop; ++idx) {
            __hacl_details__::__hacl_intrinsic_vec_dup(
                uplo_matrix + idx * 64, //dst
                float(0.0), //src
                block_size, // repeat times
                block_size * 2 / 8, // dst repeat stride
                1 // dst block stride
            );
        }
        if (remain) {
            __hacl_details__::__hacl_intrinsic_move_mask(remain);
            __hacl_details__::__hacl_intrinsic_vec_dup(
                uplo_matrix + 64 * loop, //dst
                float(0.0), //src
                block_size, // repeat times
                block_size * 2 / 8, // dst repeat stride
                1 // dst block stride
            );
        }
        for (int i = 0; i < block_size; ++i) {
            vec_dup(uplo_matrix + block_size * 2 * i, float(1.0), 2 * (i + 1));
        }
    } else {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        for (int idx = 0; idx < loop; ++idx) {
            __hacl_details__::__hacl_intrinsic_vec_dup(
                uplo_matrix + idx * 64, //dst
                float(1.0), //src
                block_size, // repeat times
                block_size * 2 / 8, // dst repeat stride
                1 // dst block stride
            );
        }
        if (remain) {
            __hacl_details__::__hacl_intrinsic_move_mask(remain);
            __hacl_details__::__hacl_intrinsic_vec_dup(
                uplo_matrix + 64 * loop, //dst
                float(1.0), //src
                block_size, // repeat times
                block_size * 2 / 8, // dst repeat stride
                1 // dst block stride
            );
        }
        for (int i = 0; i < block_size; ++i) {
            set_flag(PIPE_S, PIPE_V, 3);
            wait_flag(PIPE_S, PIPE_V, 3);
            vec_dup(uplo_matrix + block_size * 2 * i, float(0.0), 2 * (i + 1));
            set_flag(PIPE_V, PIPE_S, 3);
            wait_flag(PIPE_V, PIPE_S, 3);
            *(uplo_matrix + block_size * 2 * i + 2 * i) = 1.0;
            *(uplo_matrix + block_size * 2 * i + 2 * i + 1) = 1.0;
        }
    }
}


extern "C" __global__ __aicore__ void hablas_ctrmv_kernel(int64_t uplo,
                                                  int64_t trans,
                                                  int64_t diag,
                                                  int64_t M,
                                                  __gm__ float *A,
                                                  int64_t lda,
                                                  __gm__ float *X,
                                                  int64_t incx,
                                                  __gm__ float *workspace,
                                                  __gm__ float *tmp_gm,
                                                  int64_t base_block_size)
// extern "C" __global__ __aicore__ void hablas_ctrmv_kernel(
//                                         __gm__ float *A,
//                                         __gm__ float *X,
//                                         __gm__ float *tmp_gm,
//                                         __gm__ float *workspace)
{
    // int64_t uplo = 1;
    // int64_t trans = 0;
    // int64_t diag = 0;
    // int64_t M = 320;
    // int64_t lda = 320;
    // int64_t incx = 1;
    
    Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_a_block_real;
    Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_a_block_imag;
    // Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_a_block_separate_real;
    // Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_a_block_separate_imag;

    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_x_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_x_block_imag;
    // Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_x_block_separate_real;
    // Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_x_block_separate_imag;

    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_res_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_res_block_imag;
    // Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_res_block_separate_real;
    // Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_res_block_separate_imag;

    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_separate_vector;

    Vector<float_8, UB_TMP_BLOCK_SIZE / 8, HACL_UB> ub_tmp_block;
    Vector<float_8, UB_WORKSPACE_SIZE / 8, HACL_UB> ub_wksp_block;

    Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_fill_zero_block;


    __ub__ float *ub_a_block_real_ptr = ub_a_block_real.get_ptr(0);
    __ub__ float *ub_a_block_imag_ptr = ub_a_block_imag.get_ptr(0);
    // __ub__ float *ub_a_block_separate_real_ptr = ub_a_block_separate_real.get_ptr(0);
    // __ub__ float *ub_a_block_separate_imag_ptr = ub_a_block_separate_imag.get_ptr(0);

    __ub__ float *ub_x_block_real_ptr = ub_x_block_real.get_ptr(0);
    __ub__ float *ub_x_block_imag_ptr = ub_x_block_imag.get_ptr(0);
    // __ub__ float *ub_x_block_separate_real_ptr = ub_x_block_separate_real.get_ptr(0);
    // __ub__ float *ub_x_block_separate_imag_ptr = ub_x_block_separate_imag.get_ptr(0);
    
    __ub__ float *ub_res_block_real_ptr = ub_res_block_real.get_ptr(0);
    __ub__ float *ub_res_block_imag_ptr = ub_res_block_imag.get_ptr(0);
    // __ub__ float *ub_res_block_separate_real_ptr = ub_res_block_separate_real.get_ptr(0);
    // __ub__ float *ub_res_block_separate_imag_ptr = ub_res_block_separate_imag.get_ptr(0);

    __ub__ float *ub_separate_vector_ptr = ub_separate_vector.get_ptr(0);


    // 中间存储空间 用来存储转置情况下的中间计算结果
    __ub__ float *ub_tmp_block_ptr = ub_tmp_block.get_ptr(0);
    // 中间存储空间 用来搬运向量
    __ub__ float *ub_wksp_block_ptr = ub_wksp_block.get_ptr(0);
    // 三角置零矩阵
    __ub__ float *ub_fill_zero_block_ptr = ub_fill_zero_block.get_ptr(0);

    hablas_generate_uplo_matrix(ub_fill_zero_block_ptr, uplo);

pipe_barrier(PIPE_ALL);
    vec_dup(ub_separate_vector_ptr, float(0), UB_VECTOR_SIZE * 2);

pipe_barrier(PIPE_ALL);
    for (int idx = 0; idx < UB_VECTOR_SIZE * 2; idx += 2) {
        *(ub_separate_vector_ptr + idx) = 1;
    }

pipe_barrier(PIPE_ALL);

    int64_t m = base_block_size; // 基块大小

    int64_t m_tiles  = (M + m - 1) / m ;
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
    set_flag(PIPE_V, PIPE_MTE2, 1);
    set_flag(PIPE_V, PIPE_MTE2, 2);
    set_flag(PIPE_V, PIPE_MTE2, 3);
    set_flag(PIPE_MTE2, PIPE_MTE3, 0);
    set_flag(PIPE_MTE3, PIPE_V, 0); 


    for (int64_t tiles_idx = 0; tiles_idx < tiles_per_core; ++tiles_idx) {
        int64_t block_index = tiles_idx * block_num + block_idx;
        int64_t row = block_index / n_tiles;
        int64_t m_real = m;
        if (row == m_tiles - 1 && m_remain > 0) {
            m_real = m_remain;
        }
        int64_t m_real_pad = m_real % 8 ? (m_real & 0xfffffff8) + 8 : m_real; 

        __gm__ float *wk_ptr = workspace + row * m * 2;
        __gm__ float *tmp_gm_ptr = tmp_gm + row * m * 2;


        // uplo = 1上三角矩阵 
        // 矩阵A在右边 向量X在左边 启用GEMV模式
        int32_t k_idx = row;
        int32_t k_dst = k_loop;
        if (trans - uplo == 0) {
            k_idx = 0;
            k_dst = row + 1;
        } 
                 
        for (; k_idx < k_dst; ++k_idx) {
            int32_t k_real = m;
            if (k_idx == k_loop - 1 && k_remain > 0) {
                k_real = k_remain;
            }
            int64_t k_real_pad = k_real % 8 ? (k_real & 0xfffffff8) + 8 : k_real;
            __gm__ float *X_ptr = X + m * incx * k_idx * 2;

            if (trans == 0) {
                __gm__ float *A_ptr = A + m * row * 2 + k_idx * m * lda * 2;

                wait_flag(PIPE_V, PIPE_MTE2, 0);// waiting for ub_a_block_real_ptr
                hablas_load_cmatrix_gm2ub(ub_a_block_real_ptr, A_ptr, m_real, k_real, m_real_pad, k_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_V, 0);

                wait_flag(PIPE_MTE2, PIPE_V, 0);
                hablas_complex_to_real_imag(ub_a_block_real_ptr, ub_a_block_real_ptr, ub_separate_vector_ptr, m_real_pad, k_real_pad, m_real, k_real);
                
                wait_flag(PIPE_V, PIPE_MTE2, 1);// waiting for ub_a_block_imag_ptr
                hablas_load_cmatrix_gm2ub(ub_a_block_imag_ptr, A_ptr + 1, m_real, k_real, m_real_pad, k_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_V, 1);

                wait_flag(PIPE_MTE2, PIPE_V, 1);
                hablas_complex_to_real_imag(ub_a_block_imag_ptr, ub_a_block_imag_ptr, ub_separate_vector_ptr, m_real_pad, k_real_pad, m_real, k_real);

                if (k_idx == row) {
                    hablas_fill_zero(ub_a_block_real_ptr, uplo, diag, m_real, m_real_pad, 0, ub_fill_zero_block_ptr);
                    hablas_fill_zero(ub_a_block_imag_ptr, uplo, diag, m_real, m_real_pad, 1, ub_fill_zero_block_ptr);
                }

                wait_flag(PIPE_V, PIPE_MTE2, 2);// waiting for ub_x_block_real_ptr
                hablas_load_cvector_gm2ub(ub_x_block_real_ptr, X_ptr, ub_wksp_block_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, 2);

                wait_flag(PIPE_MTE2, PIPE_V, 2);
                hablas_complex_to_real_imag(ub_x_block_real_ptr, ub_x_block_real_ptr, ub_separate_vector_ptr, k_real_pad, 1, k_real, 1);

                wait_flag(PIPE_V, PIPE_MTE2, 3); // waiting for ub_x_block_imag_ptr
                hablas_load_cvector_gm2ub(ub_x_block_imag_ptr, X_ptr + 1, ub_wksp_block_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, 3);

                wait_flag(PIPE_MTE2, PIPE_V, 3);
                hablas_complex_to_real_imag(ub_x_block_imag_ptr, ub_x_block_imag_ptr, ub_separate_vector_ptr, k_real_pad, 1, k_real, 1);


                if (k_idx == 0 || (((trans - uplo) != 0) && k_idx == row)) {
                    wait_flag(PIPE_MTE3, PIPE_V, 0); // waiting for ub_res_block_real_ptr
                    vec_dup(ub_res_block_real_ptr, (float)0, m * 2);
                    vec_dup(ub_res_block_imag_ptr, (float)0, m * 2);
                }

                set_flag(PIPE_V, PIPE_S, 0);
                wait_flag(PIPE_V, PIPE_S, 0);

                hablas_complex_muls_notrans(ub_res_block_real_ptr,
                                            ub_res_block_imag_ptr,
                                            ub_a_block_real_ptr,
                                            ub_a_block_imag_ptr,
                                            ub_x_block_real_ptr,
                                            ub_x_block_imag_ptr,
                                            m_real * 2, k_real,
                                            m_real_pad * 2, k_real_pad);
                set_flag(PIPE_V, PIPE_MTE2, 0); // ub_a_block_real_ptr done
                set_flag(PIPE_V, PIPE_MTE2, 1); // ub_a_block_imag_ptr done
                set_flag(PIPE_V, PIPE_MTE2, 2); // ub_x_block_real_ptr done
                set_flag(PIPE_V, PIPE_MTE2, 3); // ub_x_block_imag_ptr done
            } else {
                __gm__ float *A_ptr = A + m * row * lda * 2 + k_idx * m * 2;
            
                wait_flag(PIPE_V, PIPE_MTE2, 0);// waiting for ub_a_block_real_ptr
                hablas_load_cmatrix_gm2ub(ub_a_block_real_ptr, A_ptr, k_real, m_real, k_real_pad, m_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_V, 0);

                wait_flag(PIPE_MTE2, PIPE_V, 0);
                hablas_complex_to_real_imag(ub_a_block_real_ptr, ub_a_block_real_ptr, ub_separate_vector_ptr, k_real_pad, m_real_pad, k_real, m_real);

                wait_flag(PIPE_V, PIPE_MTE2, 1);// waiting for ub_a_block_imag_ptr
                hablas_load_cmatrix_gm2ub(ub_a_block_imag_ptr, A_ptr + 1, k_real, m_real, k_real_pad, m_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_V, 1);

                wait_flag(PIPE_MTE2, PIPE_V, 1);
                hablas_complex_to_real_imag(ub_a_block_imag_ptr, ub_a_block_imag_ptr, ub_separate_vector_ptr, k_real_pad, m_real_pad, k_real, m_real);

                if (k_idx == row) {
                    hablas_fill_zero(ub_a_block_real_ptr, uplo, diag, m_real, m_real_pad, 0, ub_fill_zero_block_ptr);
                    hablas_fill_zero(ub_a_block_imag_ptr, uplo, diag, m_real, m_real_pad, 1, ub_fill_zero_block_ptr);
                }

                wait_flag(PIPE_V, PIPE_MTE2, 2);// waiting for ub_x_block_real_ptr
                hablas_load_cvector_gm2ub(ub_x_block_real_ptr, X_ptr, ub_wksp_block_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, 2);

                wait_flag(PIPE_MTE2, PIPE_V, 2);

                hablas_complex_to_real_imag(ub_x_block_real_ptr, ub_x_block_real_ptr, ub_separate_vector_ptr, k_real_pad, 1, k_real, 1);

                wait_flag(PIPE_V, PIPE_MTE2, 3); // waiting for ub_x_block_imag_ptr
                hablas_load_cvector_gm2ub(ub_x_block_imag_ptr, X_ptr + 1, ub_wksp_block_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, 3);

                wait_flag(PIPE_MTE2, PIPE_V, 3);
                hablas_complex_to_real_imag(ub_x_block_imag_ptr, ub_x_block_imag_ptr, ub_separate_vector_ptr, k_real_pad, 1, k_real, 1);

                if (k_idx == 0 || (((trans - uplo) != 0) && k_idx == row)) {
                    wait_flag(PIPE_MTE3, PIPE_V, 0); // waiting for ub_res_block_real_ptr
                    vec_dup(ub_res_block_real_ptr, (float)0, m * 2);
                    vec_dup(ub_res_block_imag_ptr, (float)0, m * 2);
                }
                hablas_complex_muls_trans(ub_res_block_real_ptr,
                                          ub_res_block_imag_ptr,
                                          ub_a_block_real_ptr,
                                          ub_a_block_imag_ptr,
                                          ub_x_block_real_ptr,
                                          ub_x_block_imag_ptr,
                                          ub_tmp_block_ptr,
                                          m_real, k_real,
                                          m_real_pad, k_real_pad);
                
                set_flag(PIPE_V, PIPE_MTE2, 0); // ub_a_block_real_ptr done
                set_flag(PIPE_V, PIPE_MTE2, 1); // ub_a_block_imag_ptr done
                set_flag(PIPE_V, PIPE_MTE2, 2); // ub_x_block_real_ptr done
                set_flag(PIPE_V, PIPE_MTE2, 3); // ub_x_block_imag_ptr done
            }
            if (k_idx == k_dst - 1) {
                set_flag(PIPE_V, PIPE_MTE3, 0);
            }
        }
        wait_flag(PIPE_V, PIPE_MTE3, 0);

        wait_flag(PIPE_MTE2, PIPE_MTE3, 0);// waiting for tmp_gm_ptr
        set_flag(PIPE_V, PIPE_S, 0);
        wait_flag(PIPE_V, PIPE_S, 0);
        hablas_memcpy(tmp_gm_ptr, ub_res_block_imag_ptr, m_real * 2, M * 2);
        set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);

        _memcpy(ub_res_block_imag_ptr, tmp_gm_ptr - 1, m_real * 2);
        set_flag(PIPE_MTE2, PIPE_MTE3, 0); // tmp_gm_ptr done

        set_flag(PIPE_MTE2, PIPE_S, 0);
        wait_flag(PIPE_MTE2, PIPE_S, 0);
        
        *(ub_res_block_imag_ptr) = 0.0;
        set_flag(PIPE_S, PIPE_V, 0);
        wait_flag(PIPE_S, PIPE_V, 0);

        vec_add(ub_res_block_real_ptr, ub_res_block_imag_ptr, ub_res_block_real_ptr, m_real * 2);
        set_flag(PIPE_V, PIPE_MTE3, 0);
        wait_flag(PIPE_V, PIPE_MTE3, 0);
        set_flag(PIPE_V, PIPE_S, 0);
        wait_flag(PIPE_V, PIPE_S, 0);
        hablas_memcpy(wk_ptr, ub_res_block_real_ptr, m_real * 2, M * 2);
        set_flag(PIPE_MTE3, PIPE_V, 0); // waiting for ub_res_block_real_ptr
    }
    
    wait_flag(PIPE_V, PIPE_MTE2, 0);
    wait_flag(PIPE_V, PIPE_MTE2, 1);
    wait_flag(PIPE_V, PIPE_MTE2, 2);
    wait_flag(PIPE_V, PIPE_MTE2, 3);
    wait_flag(PIPE_MTE2, PIPE_MTE3, 0);
    wait_flag(PIPE_MTE3, PIPE_V, 0);
}