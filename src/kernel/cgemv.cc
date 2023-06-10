#include <hacl/hacl.h>

typedef struct {
    float a;
    float b; 
} float2;

typedef float2 T;
typedef float HT;

constexpr int64_t BLOCK_DIM = 96;
constexpr int64_t UB_MATRIX_SIZE = BLOCK_DIM * BLOCK_DIM;
constexpr int64_t UB_VECTOR_SIZE = BLOCK_DIM;
constexpr int64_t UB_TMP_BLOCK_SIZE = BLOCK_DIM * 64 * 2; // 用于乘法运算存储中间结果
constexpr int64_t UB_WORKSPACE_SIZE = BLOCK_DIM * BLOCK_DIM; // 用于搬运向量gm到ub inc不为1的情况 存储中间搬运结果 

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
        int64_t content = UB_WORKSPACE_SIZE;
        int64_t loop = valid_len * inc * 2 / content;
        int64_t remain = valid_len * inc * 2 % content;
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
                           __ub__ float *wksp,
                          int64_t valid_len, 
                          int64_t inc) 
{
    if (inc == 1) {
        _memcpy(dst, src, valid_len * 2);
    } else {
        int64_t load_data_num = UB_WORKSPACE_SIZE /(inc*2)*(inc*2);
        if (!load_data_num) ++load_data_num;
        int64_t loop = (valid_len*inc*2 + load_data_num -1) / load_data_num;
        int64_t remain = (valid_len * inc * 2)%load_data_num;
        for (int i = 0; i < loop; ++i) {
            int64_t real_len = load_data_num;
            if (i+1==loop && remain != 0)
                real_len = remain;
            pipe_barrier(PIPE_ALL);
            _memcpy(wksp, dst+i*load_data_num, real_len);
            pipe_barrier(PIPE_ALL);
            for (int j = 0; j < real_len/(inc*2); ++j) {
                wksp[j*inc*2] = src[i*load_data_num/inc + j*2];
                wksp[j*inc*2+1] = src[i*load_data_num/inc + j*2+1];
            }
            pipe_barrier(PIPE_ALL);
            _memcpy(dst+i*load_data_num, wksp, real_len);
            pipe_barrier(PIPE_ALL);
        }
    }
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

HACL_INLINE __aicore__ void
hablas_complex_muls_alpha(__ub__ float *real_dst,
                            __ub__ float *imag_dst,
                            __ub__ float *real_src,
                            __ub__ float *imag_src,
                            T alpha,
                            int64_t vaild_len) 
{
    pipe_barrier(PIPE_ALL);
    _memcpy(real_src, real_dst, vaild_len);
    _memcpy(imag_src, imag_dst, vaild_len);
    vec_muls(real_dst, real_src, alpha.a, vaild_len);
    vec_axpy(real_dst, imag_src, -alpha.b, vaild_len);
    vec_muls(imag_dst, real_src, alpha.b, vaild_len);
    vec_axpy(imag_dst, imag_src, alpha.a, vaild_len);
    pipe_barrier(PIPE_ALL);
}

extern "C" __global__ __aicore__ void hablas_cgemv_kernel(
	int64_t trans,
	int64_t M,
	int64_t N,
	__gm__ float *alpha_i,
	__gm__ float *A,
	int64_t lda,
	__gm__ float *X,
	int64_t incx,
	__gm__ float *beta_i,
	__gm__ float *Y,
	int64_t incy,
    __gm__ float *tmp_gm)
{
	Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_a_block_real;
    Vector<float_8, UB_MATRIX_SIZE / 8 * 2, HACL_UB> ub_a_block_imag;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_x_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_x_block_imag;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_y_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_y_block_imag;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_buf_block_real;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_buf_block_imag;
    // Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_res_block_real;
    // Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_res_block_imag;
    Vector<float_8, UB_VECTOR_SIZE / 8 * 2, HACL_UB> ub_separate_vector;
    Vector<float_8, UB_TMP_BLOCK_SIZE / 8, HACL_UB> ub_tmp_block;
    Vector<float_8, UB_WORKSPACE_SIZE / 8, HACL_UB> ub_wksp_block;
    __ub__ float *ub_a_block_real_ptr = ub_a_block_real.get_ptr(0);
    __ub__ float *ub_a_block_imag_ptr = ub_a_block_imag.get_ptr(0);
    __ub__ float *ub_x_block_real_ptr = ub_x_block_real.get_ptr(0);
    __ub__ float *ub_x_block_imag_ptr = ub_x_block_imag.get_ptr(0);
    __ub__ float *ub_y_block_real_ptr = ub_y_block_real.get_ptr(0);
    __ub__ float *ub_y_block_imag_ptr = ub_y_block_imag.get_ptr(0);
    // __ub__ float *ub_res_block_real_ptr = ub_res_block_real.get_ptr(0);
    // __ub__ float *ub_res_block_imag_ptr = ub_res_block_imag.get_ptr(0);
    __ub__ float *ub_separate_vector_ptr = ub_separate_vector.get_ptr(0);
    __ub__ float *ub_tmp_block_ptr = ub_tmp_block.get_ptr(0);
    __ub__ float *ub_tmp_block_real_ptr = ub_buf_block_real.get_ptr(0);
    __ub__ float *ub_tmp_block_imag_ptr = ub_buf_block_imag.get_ptr(0);
    // 中间存储空间 用来搬运向量
    __ub__ float *ub_wksp_block_ptr = ub_wksp_block.get_ptr(0);
pipe_barrier(PIPE_ALL);
    vec_dup(ub_separate_vector_ptr, float(0), UB_VECTOR_SIZE * 2);
pipe_barrier(PIPE_ALL);
    for (int idx = 0; idx < UB_VECTOR_SIZE * 2; idx += 2) {
        *(ub_separate_vector_ptr + idx) = 1;
    }
pipe_barrier(PIPE_ALL);
    int64_t m = BLOCK_DIM; // 基块大小

    int64_t m_tiles  = (M + m - 1) / m;
    int64_t n_tiles  = 1;
    int64_t k_loop   = (N + m - 1) / m;
    int64_t m_remain = M % m;
    int64_t k_remain = N % m;
    if (trans != 0) {
        m_tiles  = (N + m - 1) / m;
        n_tiles  = 1;
        k_loop   = (M + m - 1) / m;
        m_remain = N % m;
        k_remain = M % m;
    }

    _memcpy(ub_wksp_block_ptr, alpha_i, 2);
    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0);
    T alpha, beta;

    alpha.a = *(ub_wksp_block_ptr);
    alpha.b = *(ub_wksp_block_ptr + 1);

    set_flag(PIPE_S, PIPE_MTE2, 0);
    wait_flag(PIPE_S, PIPE_MTE2, 0);
    _memcpy(ub_wksp_block_ptr, beta_i, 2);

    set_flag(PIPE_MTE2, PIPE_S, 0);
    wait_flag(PIPE_MTE2, PIPE_S, 0); 
    beta.a = *(ub_wksp_block_ptr);
    beta.b = *(ub_wksp_block_ptr + 1);

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

        __gm__ float *Y_ptr = Y + row * m * incy * 2;
        __gm__ float *tmp_gm_ptr = tmp_gm + row * m * 2+1;

        wait_flag(PIPE_MTE3, PIPE_V, 0); 
        // vec_dup(ub_y_block_real_ptr, (float)0, m * 2);
        // vec_dup(ub_y_block_imag_ptr, (float)0, m * 2);
        wait_flag(PIPE_V, PIPE_MTE2, 2);// waiting for ub_y_block_real_ptr
        hablas_load_cvector_gm2ub(ub_y_block_real_ptr, Y_ptr, ub_wksp_block_ptr, m_real, incy);
        set_flag(PIPE_MTE2, PIPE_V, 2);
        set_flag(PIPE_V, PIPE_MTE2, 2);

        wait_flag(PIPE_MTE2, PIPE_V, 2);
        hablas_complex_to_real_imag(ub_y_block_real_ptr, ub_y_block_real_ptr, ub_separate_vector_ptr, m_real_pad, 1, m_real, 1);

        wait_flag(PIPE_V, PIPE_MTE2, 3); // waiting for ub_x_block_imag_ptr
        hablas_load_cvector_gm2ub(ub_y_block_imag_ptr, Y_ptr + 1, ub_wksp_block_ptr, m_real, incy);
        set_flag(PIPE_MTE2, PIPE_V, 3);
        set_flag(PIPE_V, PIPE_MTE2, 3);

        wait_flag(PIPE_MTE2, PIPE_V, 3);
        hablas_complex_to_real_imag(ub_y_block_imag_ptr, ub_y_block_imag_ptr, ub_separate_vector_ptr, m_real_pad, 1, m_real, 1);
        hablas_complex_muls_alpha(ub_y_block_real_ptr, ub_y_block_imag_ptr, ub_tmp_block_real_ptr, ub_tmp_block_imag_ptr, beta, m_real*2);
        // uplo = 1上三角矩阵 
        // 矩阵A在右边 向量X在左边 启用GEMV模式
                 
        for (int64_t k_idx = 0; k_idx < k_loop; ++k_idx) {
            int64_t k_real = m;
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
                hablas_complex_muls_alpha(ub_x_block_real_ptr, ub_x_block_imag_ptr, ub_tmp_block_real_ptr, ub_tmp_block_imag_ptr, alpha, k_real*2);

                // if (k_idx == 0) {
                //     wait_flag(PIPE_MTE3, PIPE_V, 0); // waiting for ub_res_block_real_ptr
                //     vec_dup(ub_res_block_real_ptr, (float)0, m * 2);
                //     vec_dup(ub_res_block_imag_ptr, (float)0, m * 2);
                // }

                set_flag(PIPE_V, PIPE_S, 0);
                wait_flag(PIPE_V, PIPE_S, 0);

                hablas_complex_muls_notrans(ub_y_block_real_ptr,
                                            ub_y_block_imag_ptr,
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
                if (trans == 2) {
                    vec_muls(ub_a_block_imag, ub_a_block_imag, -1.0f);
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
                hablas_complex_muls_alpha(ub_x_block_real_ptr, ub_x_block_imag_ptr, ub_tmp_block_real_ptr, ub_tmp_block_imag_ptr, alpha, k_real*2);
                // if (k_idx == 0) {
                //     wait_flag(PIPE_MTE3, PIPE_V, 0); // waiting for ub_res_block_real_ptr
                //     vec_dup(ub_res_block_real_ptr, (float)0, m * 2);
                //     vec_dup(ub_res_block_imag_ptr, (float)0, m * 2);
                // }
                hablas_complex_muls_trans(ub_y_block_real_ptr,
                                          ub_y_block_imag_ptr,
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
            if (k_idx == k_loop - 1) {
                set_flag(PIPE_V, PIPE_MTE3, 0);
            }
        }
        wait_flag(PIPE_V, PIPE_MTE3, 0);



        wait_flag(PIPE_MTE2, PIPE_MTE3, 0);// waiting for tmp_gm_ptr
        _memcpy(tmp_gm_ptr, ub_y_block_imag_ptr, m_real_pad * 2);
        set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);

        _memcpy(ub_y_block_imag_ptr, tmp_gm_ptr - 1, m_real * 2);
        set_flag(PIPE_MTE2, PIPE_MTE3, 0); // tmp_gm_ptr done

        set_flag(PIPE_MTE2, PIPE_S, 0);
        wait_flag(PIPE_MTE2, PIPE_S, 0);
        
        *(ub_y_block_imag_ptr) = 0.0;
        set_flag(PIPE_S, PIPE_V, 0);
        wait_flag(PIPE_S, PIPE_V, 0);

        vec_add(ub_y_block_real_ptr, ub_y_block_imag_ptr, ub_y_block_real_ptr, m_real * 2);
        set_flag(PIPE_V, PIPE_MTE3, 0);
        wait_flag(PIPE_V, PIPE_MTE3, 0);

        // _memcpy(Y_ptr, ub_y_block_real_ptr, m_real * 2);
        hablas_store_cvector_ub2gm(Y_ptr, ub_y_block_real_ptr, ub_wksp_block_ptr, m_real_pad, incy);
        set_flag(PIPE_MTE3, PIPE_V, 0); // waiting for ub_res_block_real_ptr
    }
    
    wait_flag(PIPE_V, PIPE_MTE2, 0);
    wait_flag(PIPE_V, PIPE_MTE2, 1);
    wait_flag(PIPE_V, PIPE_MTE2, 2);
    wait_flag(PIPE_V, PIPE_MTE2, 3);
    wait_flag(PIPE_MTE2, PIPE_MTE3, 0);
    wait_flag(PIPE_MTE3, PIPE_V, 0);
}