#include <hacl/hacl.h>

constexpr int UB_MATRIX_SIZE = 128 * 128;
constexpr int UB_VECTOR_SIZE = 128;
constexpr int UB_TMP_BLOCK_SIZE = 128 * 64; // 用于乘法运算存储中间结果
constexpr int UB_WORKSPACE_SIZE = 128 * 128; // 用于搬运向量gm到ub inc不为1的情况 存储中间搬运结果 

HACL_INLINE __aicore__ void hablas_memcpy(__gm__ float *dst, __ub__ float *src, int64_t len, int64_t space) {
    if (space < 8) {
        __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(dst, src, 1, 1, 0, 0);
    } else {
        _memcpy(dst, src, len);
    }
}

HACL_INLINE __aicore__ void 
hablas_load_cmatrix_gm2l1(__l1__ float *dst,
                          __gm__ float *src,
                          int64_t m_real,
                          int64_t n_real,
                          int64_t m_real_pad,
                          int64_t n_real_pad,
                          int64_t stride) {
    if (m_real % 8 || (stride - m_real) % 8) {
        for (int j = 0; j < n_real; ++j) {
            _memcpy(dst + j * m_real_pad , src + j * stride , 1, m_real_pad / 8 , 0, 0);
        }
    }
    else {
        _memcpy(dst, src, n_real, m_real / 8 , 0, (stride - m_real) / 8 );
    }
}

HACL_INLINE __aicore__ void 
hablas_load_cmatrix_l12ub(__ub__ float *dst, 
                          __l1__ float *src, 
                          int64_t m_real_pad, 
                          int64_t n_real_pad) {

    _memcpy(dst, src, 1, m_real_pad * n_real_pad / 8 , 0, 0);
}

HACL_INLINE __aicore__ void 
hablas_load_cvector_gm2ub(__ub__ float *dst, 
                          __gm__ float *src, 
                          __ub__ float *wksp,
                          int64_t valid_len, 
                          int64_t inc) 
{
    if (inc == 1) {
        _memcpy(dst, src, valid_len);
    } else {
        int32_t content = UB_WORKSPACE_SIZE;
        int32_t loop = valid_len * inc  / content;
        int32_t remain = valid_len * inc  % content;
        int32_t start_posi = 0;
        int32_t iub = 0;
        set_flag(PIPE_S, PIPE_MTE2, 2);
        for (int i = 0; i < loop; ++i) {
            wait_flag(PIPE_S, PIPE_MTE2, 2);
            _memcpy(wksp, 
                    src + i * content,
                    content);
            set_flag(PIPE_MTE2, PIPE_S, 2);
            wait_flag(PIPE_MTE2, PIPE_S, 2);

            int iwhile = start_posi;
            while (iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                iwhile = iwhile + inc;
                iub = iub + 1;
            }
            set_flag(PIPE_S, PIPE_MTE2, 2);
            start_posi = iwhile - content;
        }
        if (remain) {
            wait_flag(PIPE_S, PIPE_MTE2, 2);
            _memcpy(wksp, 
                    src + loop * content,
                    remain);
            set_flag(PIPE_MTE2, PIPE_S, 2);
            wait_flag(PIPE_MTE2, PIPE_S, 2);
            int iwhile = start_posi;
            while (iub < valid_len && iwhile < content) {
                *(dst + iub) = *(wksp + iwhile);
                iwhile = iwhile + inc;
                iub = iub + 1;
            }
            set_flag(PIPE_S, PIPE_MTE2, 2);
        }
        wait_flag(PIPE_S, PIPE_MTE2, 2);
    }
}

HACL_INLINE __aicore__ void 
hablas_store_cvector_ub2gm(__gm__ float *dst, 
                           __ub__ float *src, 
                           int64_t valid_len) 
{
    _memcpy(dst, src, valid_len);
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
        float t = *(src1 + n_idx);
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
    int64_t loop = n_real / 64;
    int64_t remain = n_real % 64;
    for (int64_t idx = 0; idx < loop; ++idx) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_mla(tmp,
                                 src0 + 64 * idx,
                                 src1 + 64 * idx,
                                 m_real,//repeat times
                                 8,// dst repeat stride
                                 n_real_pad / 8, // src0 repeat stride
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
                                 8,// dst repeat stride
                                 n_real_pad / 8, // src0 repeat stride
                                 0,// src1 repeat stride
                                 1,// dst block stride
                                 1,// src0 block stride
                                 1// src1 block stride
                                 );
    }
    __hacl_details__::__hacl_intrinsic_move_mask(64);
    __hacl_details__::__hacl_intrinsic_vec_reduce_add(tmp,
                                    tmp,
                                    m_real, //repeat times
                                    8, //src repeat stide
                                    1  //src block stride
                                    );
    if (flag) {
        vec_sub(dst, dst, tmp, m_real);
    } else {
        vec_add(dst, dst, tmp, m_real);
    }
}

HACL_INLINE __aicore__ void
hablas_complex_muls_trans(__ub__ float *real_dst,
                            __ub__ float *real_src0,
                            __ub__ float *real_src1,
                            __ub__ float *tmp,
                            int64_t m_real,
                            int64_t n_real,
                            int64_t m_real_pad,
                            int64_t n_real_pad) 
{
    hablas_matrix_vector_muls_trans(real_dst, real_src0, real_src1, tmp, m_real, n_real, m_real_pad, n_real_pad, 0);
}

HACL_INLINE __aicore__ void
hablas_complex_muls_notrans(__ub__ float *real_dst,
                            __ub__ float *real_src0,
                            __ub__ float *real_src1,
                            int64_t m_real,
                            int64_t n_real,
                            int64_t m_real_pad,
                            int64_t n_real_pad) 
{
    hablas_matrix_vector_muls_notrans(real_dst, real_src0, real_src1, m_real, n_real, m_real_pad, n_real_pad, 0);
}

HACL_INLINE __aicore__ void hablas_fill_zero(__ub__ float *dst,
                                             int32_t uplo,
                                             int32_t diag,
                                             int32_t m_real,
                                             int32_t m_real_pad,
                                             __ub__ float *uplo_matrix)
{
    int64_t loop = m_real / 64;
    int64_t remain = m_real % 64;

    for (int loop_idx = 0; loop_idx < loop; ++loop_idx) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_mul<float>(
            dst + loop_idx * 64,
            dst + loop_idx * 64,
            uplo_matrix + loop_idx * 64,
            m_real,// repeat times
            m_real_pad / 8, // dst repeat stride
            m_real_pad / 8, // src0 repeat stride
            128 / 8, // src1 repeat stride
            1,// dst block stride
            1,// src0 block stride
            1// src1 block stride
        );
    }
    if (remain) {
        __hacl_details__::__hacl_intrinsic_move_mask(remain);
        __hacl_details__::__hacl_intrinsic_vec_mul<float>(
            dst + loop * 64,
            dst + loop * 64,
            uplo_matrix + loop * 64,
            m_real,// repeat times
            m_real_pad / 8, // dst repeat stride
            m_real_pad / 8, // src0 repeat stride
            128 / 8, // src1 repeat stride
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
                *(dst + i * m_real_pad + j) = 1.0;
            }
        }
    }
    set_flag(PIPE_S, PIPE_V, 3);
    wait_flag(PIPE_S, PIPE_V, 3);
}


extern "C" __global__ __aicore__ void hablas_strmv_kernel(int64_t uplo,
                                                  int64_t trans,
                                                  int64_t diag,
                                                  int64_t M,
                                                  __gm__ float *A,
                                                  int64_t lda,
                                                  __gm__ float *X,
                                                  int64_t incx,
                                                  __gm__ float *workspace,
                                                  int64_t base_block_size)

// extern "C" __global__ __aicore__ void hablas_strmv_kernel(
//                                                   __gm__ float *A,
//                                                   __gm__ float *X,
//                                                   __gm__ float *workspace)
{

    // int64_t uplo = 1;
    // int64_t trans = 0;
    // int64_t diag = 0;
    // int64_t M = 1024;
    // int64_t lda = 1024;
    // int64_t incx = 1;

    Vector<float_8, UB_MATRIX_SIZE / 8, HACL_UB> ub_a_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8, HACL_UB> ub_x_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8, HACL_UB> ub_res_block_real;
    Vector<float_8, UB_TMP_BLOCK_SIZE / 8, HACL_UB> ub_tmp_block;
    Vector<float_8, UB_WORKSPACE_SIZE / 8, HACL_UB> ub_wksp_block;
    Vector<float_8, UB_MATRIX_SIZE / 8, HACL_UB> ub_fill_block;
    

    Vector<float_8, UB_MATRIX_SIZE / 8 , HACL_L1> l1_a_pg_block;

    __ub__ float *ub_a_block_real_ptr = ub_a_block_real.get_ptr(0);
    __ub__ float *ub_x_block_real_ptr = ub_x_block_real.get_ptr(0);
    __ub__ float *ub_res_block_real_ptr = ub_res_block_real.get_ptr(0);
    __l1__ float *l1_a_pg_block_ptr = l1_a_pg_block.get_ptr(0);

    // 中间存储空间 用来存储转置情况下的中间计算结果
    __ub__ float *ub_tmp_block_ptr = ub_tmp_block.get_ptr(0);
    // 中间存储空间 用来搬运向量
    __ub__ float *ub_wksp_block_ptr = ub_wksp_block.get_ptr(0);
    __ub__ float *ub_fill_block_ptr = ub_fill_block.get_ptr(0);

    if (uplo) {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_dup(
            ub_fill_block_ptr, //dst
            float(0.0), //src
            128, // repeat times
            128 / 8, // dst repeat stride
            1 // dst block stride
        );
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_dup(
            ub_fill_block_ptr + 64, //dst
            float(0.0), //src
            128, // repeat times
            128 / 8, // dst repeat stride
            1 // dst block stride
        );
        for (int i = 0; i < 128; ++i) {
            vec_dup(ub_fill_block_ptr + 128 * i, float(1.0), i + 1);
        }
    } else {
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_dup(
            ub_fill_block_ptr, //dst
            float(1.0), //src
            128, // repeat times
            128 / 8, // dst repeat stride
            1 // dst block stride
        );
        __hacl_details__::__hacl_intrinsic_move_mask(64);
        __hacl_details__::__hacl_intrinsic_vec_dup(
            ub_fill_block_ptr + 64, //dst
            float(1.0), //src
            128, // repeat times
            128 / 8, // dst repeat stride
            1 // dst block stride
        );
        for (int i = 0; i < 128; ++i) {
            set_flag(PIPE_S, PIPE_V, 3);
            wait_flag(PIPE_S, PIPE_V, 3);
            vec_dup(ub_fill_block_ptr + 128 * i, float(0.0), i + 1);
            set_flag(PIPE_V, PIPE_S, 3);
            wait_flag(PIPE_V, PIPE_S, 3);
            *(ub_fill_block_ptr + 128 * i + i) = 1.0;
        }
    }

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
    
    set_flag(PIPE_MTE1, PIPE_MTE2, 0); 
    set_flag(PIPE_V, PIPE_MTE1, 0);
    set_flag(PIPE_V, PIPE_MTE2, 1);

    for (int64_t tiles_idx = 0; tiles_idx < tiles_per_core; ++tiles_idx) {
        int64_t block_index = tiles_idx * block_num + block_idx;
        int64_t row = block_index / n_tiles;
        int64_t m_real = m;
        if (row == m_tiles - 1 && m_remain > 0) {
            m_real = m_remain;
        }
        int64_t m_real_pad = m_real % 8 ? (m_real & 0xfffffff8) + 8 : m_real; 

        __gm__ float *wk_ptr = workspace + row * m ;


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
            __gm__ float *X_ptr = X + m * incx * k_idx;

            if (trans == 0) {
                __gm__ float *A_ptr = A + m * row + k_idx * m * lda ;

                wait_flag(PIPE_MTE1, PIPE_MTE2, 0); // 等待l1_a_pg_block数据使用完成
                hablas_load_cmatrix_gm2l1(l1_a_pg_block_ptr, A_ptr, m_real, k_real, m_real_pad, k_real_pad, lda);
                set_flag(PIPE_MTE2, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, 0);

                wait_flag(PIPE_V, PIPE_MTE1, 0); //等待ub_a_block_real数据使用完成
                hablas_load_cmatrix_l12ub(ub_a_block_real_ptr, l1_a_pg_block_ptr, m_real_pad, k_real_pad);
                set_flag(PIPE_MTE1, PIPE_MTE2, 0); //l1_a_pg_block数据使用完成

                set_flag(PIPE_MTE1, PIPE_V, 0);
                wait_flag(PIPE_MTE1, PIPE_V, 0);


                if (k_idx == row) {
                    hablas_fill_zero(ub_a_block_real_ptr, uplo, diag, m_real, m_real_pad, ub_fill_block_ptr);
                }

                wait_flag(PIPE_V, PIPE_MTE2, 1); // 等待ub_x_block_real使用完成
                hablas_load_cvector_gm2ub(ub_x_block_real_ptr, X_ptr, ub_wksp_block_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, 1);
                wait_flag(PIPE_MTE2, PIPE_V, 1);

                if (k_idx == 0 || (((trans - uplo) != 0) && k_idx == row)) {
                    vec_dup(ub_res_block_real_ptr, (float)0, m);
                }
                set_flag(PIPE_V, PIPE_S, 2);
                wait_flag(PIPE_V, PIPE_S, 2);
                hablas_complex_muls_notrans(ub_res_block_real_ptr,
                                            ub_a_block_real_ptr,
                                            ub_x_block_real_ptr,
                                            m_real, k_real,
                                            m_real_pad, k_real_pad);
                set_flag(PIPE_V, PIPE_MTE1, 0); // ub_a_block_real使用完成
                set_flag(PIPE_V, PIPE_MTE2, 1); // ub_x_block_real使用完成

            } else {
                __gm__ float *A_ptr = A + m * row * lda  + k_idx * m ;
                wait_flag(PIPE_MTE1, PIPE_MTE2, 0); // 等待l1_a_pg_block数据使用完成
                hablas_load_cmatrix_gm2l1(l1_a_pg_block_ptr, A_ptr, k_real, m_real, k_real_pad, m_real_pad, lda);

                set_flag(PIPE_MTE2, PIPE_MTE1, 0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, 0);
                wait_flag(PIPE_V, PIPE_MTE1, 0); //等待ub_a_block_real数据使用完成

                hablas_load_cmatrix_l12ub(ub_a_block_real_ptr, l1_a_pg_block_ptr, m_real_pad, k_real_pad);

                set_flag(PIPE_MTE1, PIPE_MTE2, 0); //l1_a_pg_block数据使用完成

                set_flag(PIPE_MTE1, PIPE_V, 0);
                wait_flag(PIPE_MTE1, PIPE_V, 0);

                if (k_idx == row) {
                    hablas_fill_zero(ub_a_block_real_ptr, uplo, diag, m_real, m_real_pad, ub_fill_block_ptr);
                }

                wait_flag(PIPE_V, PIPE_MTE2, 1); // 等待ub_x_block_real使用完成
                hablas_load_cvector_gm2ub(ub_x_block_real_ptr, X_ptr, ub_wksp_block_ptr, k_real, incx);
                set_flag(PIPE_MTE2, PIPE_V, 1);
                wait_flag(PIPE_MTE2, PIPE_V, 1);
    
                if (k_idx == 0 || (((trans - uplo) != 0) && k_idx == row)) {
                    vec_dup(ub_res_block_real_ptr, (float)0, m);
                }
                hablas_complex_muls_trans(ub_res_block_real_ptr,
                                          ub_a_block_real_ptr,
                                          ub_x_block_real_ptr,
                                          ub_tmp_block_ptr,
                                          m_real, k_real,
                                          m_real_pad, k_real_pad);
                set_flag(PIPE_V, PIPE_MTE1, 0); // ub_a_block_real使用完成
                set_flag(PIPE_V, PIPE_MTE2, 1); // ub_x_block_real使用完成
            }
            if (k_idx == k_dst - 1) {
                set_flag(PIPE_V, PIPE_MTE3, 0);
            }
        }
        wait_flag(PIPE_V, PIPE_MTE3, 0);
        set_flag(PIPE_V, PIPE_S, 0);
        wait_flag(PIPE_V, PIPE_S, 0);
        hablas_memcpy(wk_ptr, ub_res_block_real_ptr, m_real, M); 
    }

    wait_flag(PIPE_MTE1, PIPE_MTE2, 0); 
    wait_flag(PIPE_V, PIPE_MTE1, 0);
    wait_flag(PIPE_V, PIPE_MTE2, 1);
}