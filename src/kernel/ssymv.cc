#include "tools.h"

const int64_t MAX_FLOAT = 64; // 90

template <typename T>
HACL_INLINE __aicore__ void memcpy_ub2gm_anylen(__gm__ T *gm, __ub__ T *ub, int64_t len)
{
    int base = 32 / sizeof(T);
    if (len < base)
        len = base;
    int64_t nUnit = len / base;
    int64_t unit_remain = len % base;
    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(gm, ub, 1, nUnit, 0, 0);
    set_flag(PIPE_MTE3, PIPE_S, 0);
    wait_flag(PIPE_MTE3, PIPE_S, 0);
    if (unit_remain && len > base)
    {
        int64_t offset_start = nUnit * base - base + unit_remain;

        for (int i = 0; i < base; ++i)
        {
            *(ub + i) = *(ub + offset_start + i);
        }
        set_flag(PIPE_S, PIPE_MTE3, 0);
        wait_flag(PIPE_S, PIPE_MTE3, 0);
        __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(gm + offset_start, ub, 1, 1, 0, 0);
    }
}

HACL_INLINE __aicore__ void compute_without_transpose(__ub__ float *block, __ub__ float *x, __ub__ float *y, int m, int n, int kernel_N)
{
    set_flag(PIPE_V, PIPE_S, 1); // set(5)
    for (int64_t i = 0; i < n; ++i)
    {
        wait_flag(PIPE_V, PIPE_S, 1); // wait(5)
        float scalar = *(x + i);
        set_flag(PIPE_S, PIPE_V, 0);
        wait_flag(PIPE_S, PIPE_V, 0);
        vec_axpy(y, block + i * kernel_N, scalar, kernel_N);
        set_flag(PIPE_V, PIPE_S, 1); // set(5)
    }
    wait_flag(PIPE_V, PIPE_S, 1); // wait(5)
}

HACL_INLINE __aicore__ void compute_transpose(__ub__ float *block, __ub__ float *x, __ub__ float *y, int m, int n, int kernel_N)
{

    Vector<float_8, 32 * MAX_FLOAT, HACL_UB> tmp1;
    tmp1 = 0.0; // 置0

    for (int64_t i = 0; i < kernel_N; i += 64)
    {
        __hacl_details__::__hacl_intrinsic_vec_mla(tmp1.get_ptr(0), block + i, x + i,
                                                   kernel_N,
                                                   8, kernel_N / 8, 0,
                                                   1, 1, 1);
    }

    Vector<float_8, MAX_FLOAT, HACL_UB> tmp2;
    __hacl_details__::__hacl_intrinsic_vec_reduce_add(tmp2.get_ptr(0), tmp1.get_ptr(0),
                                                      kernel_N, 8, 1);
    vec_add(y, tmp2.get_ptr(0), y, kernel_N);
}

HACL_INLINE __aicore__ void fillup(__ub__ float *ub, int m, int n, int64_t uplo, int kernel_N)
{
    if (uplo == 0)
    {
        for (int64_t x = 0; x < kernel_N; ++x)
        {
            for (int64_t y = x + 1; y < kernel_N; ++y)
            {
                *(ub + y * kernel_N + x) = *(ub + x * kernel_N + y);
            }
        }
    }
    else
    {
        for (int64_t x = 0; x < m; ++x)
        {
            for (int64_t y = 0; y < x; ++y)
            {
                *(ub + y * kernel_N + x) = *(ub + x * kernel_N + y);
            }
        }
    }
}

HACL_INLINE __aicore__ void load_block_gm2l1(__l1__ float *l1, __gm__ float *gm, int lda, int blockN, int m, int n)
{
    int cnt_n = n / 8;
    for (int i = 0; i < 8; ++i)
    {
        if ((i < n % 8 ? cnt_n + 1 : cnt_n) > 0)
        {
            if ((m % 8) != 0)
            {
                // 除了最后一列, 其他向后借
                int cnt_m = (m + 7) / 8;
                if (((i < n % 8) ? cnt_n + 1 : cnt_n) + (i == (n % 8 - 1) ? 0 : 0) > 0)
                    _memcpy(l1 + i * blockN, gm + i * lda,
                            (i < n % 8 ? cnt_n + 1 : cnt_n) + (i == (n % 8 - 1) ? 0 : 0),
                            cnt_m, blockN - cnt_m, lda - cnt_m);
            }
            else if(m%8 ==0)
            {
                int cnt_m = (m) / 8;
                if (cnt_m > 0)
                    _memcpy(l1 + i * blockN, gm + i * lda,
                            i < n % 8 ? cnt_n + 1 : cnt_n,
                            cnt_m, blockN - cnt_m, lda - cnt_m);
            }
        }
    }
}

HACL_INLINE __aicore__ void load_block_l12ub(__ub__ float *ub, __l1__ float *l1, int lda, int blockN, int m, int n)
{
    Vector<float_8, 64, HACL_L1> zeros;
    if (m == blockN && n == blockN)
    {
        int64_t cnt = blockN / 8;
        set_flag(PIPE_MTE1, PIPE_S, 0);
        wait_flag(PIPE_MTE1, PIPE_S, 0);
        for (int64_t i = 0; i < 8; ++i)
        {
            _memcpy(ub + i * blockN, l1 + i * blockN, cnt, blockN / 8, (7 * blockN) / 8, (7 * blockN) / 8); // 32B为单位 8个float
            pipe_barrier(PIPE_ALL);
        }
    }
    else
    {
        set_flag(PIPE_S, PIPE_MTE1, 1); // set(4)
        for (int64_t i = 0; i < n; ++i)
        {
            wait_flag(PIPE_S, PIPE_MTE1, 1); //set(4)
            _memcpy(ub + i * blockN, l1 + i * blockN, 1, (m + 7) / 8, 0, 0);
            set_flag(PIPE_MTE1, PIPE_S, 0);
            wait_flag(PIPE_MTE1, PIPE_S, 0);
            int64_t m_ = m;
            for (; (i * blockN + m_) % 8 != 0; ++m_)
            {
                *(ub + i * blockN + m_) = 0;
            }
            set_flag(PIPE_S, PIPE_MTE1, 1); //set(4)
            pipe_barrier(PIPE_ALL);
        }
        wait_flag(PIPE_S, PIPE_MTE1, 1); //wait(4)
    }
}

HACL_INLINE __aicore__ void load_x_L1(__l1__ float *l1, __gm__ float *gm, int n, int incx, int alpha)
{
    _memcpy(l1, gm, (n * incx + 7) / 8, 1, 0, 0);
}

HACL_INLINE __aicore__ void load_x_L12UB(__ub__ float *ub, __l1__ float *l1, int n, int incx, float alpha)
{

    if (incx != 1)
    {
        Vector<float_8, MAX_FLOAT, HACL_UB> buffer;

        set_flag(PIPE_S, PIPE_MTE1, 1); // set(3)
        for (int64_t i = 0; i < (n / 8); i++)
        {
            wait_flag(PIPE_S, PIPE_MTE1, 1); // wait(3)
            _memcpy(buffer.get_ptr(0), l1 + (i * incx * 8), 1, incx, 0, 0);

            set_flag(PIPE_MTE1, PIPE_S, 0);
            wait_flag(PIPE_MTE1, PIPE_S, 0);

            for (int64_t j = 0; j < 8; ++j)
            {
                *(ub + i * 8 + j) = (alpha * (*(buffer.get_ptr(0) + j * incx)));
                *(ub + i * 8 + j) = (*(buffer.get_ptr(0) + j * incx));
                *(ub + i * 8 + j) *= alpha;
            }

            set_flag(PIPE_S, PIPE_MTE1, 1); // set(3)
        }
        wait_flag(PIPE_S, PIPE_MTE1, 1); //wait(3)

        if (n % 8 != 0)
        {
            _memcpy(buffer.get_ptr(0), l1 + ((n / 8) * incx * 8), 1, incx, 0, 0);
            set_flag(PIPE_MTE1, PIPE_S, 0);
            wait_flag(PIPE_MTE1, PIPE_S, 0);

            for (int64_t j = 0; j < (n % 8); ++j)
            {
                *(ub + (n / 8) * 8 + j) = (alpha * *(buffer.get_ptr(0) + j * incx));
            }

            set_flag(PIPE_S, PIPE_MTE1, 0);
            wait_flag(PIPE_S, PIPE_MTE1, 0);
        }
    }
    else
    {
        _memcpy(ub, l1, 1, (n + 7) / 8, 0, 0);
        set_flag(PIPE_MTE1, PIPE_V, 0);
        wait_flag(PIPE_MTE1, PIPE_V, 0);
        vec_muls(ub, ub, alpha, n);

        set_flag(PIPE_V, PIPE_MTE1, 0);
        wait_flag(PIPE_V, PIPE_MTE1, 0);
    }
}



constexpr int UB_WORKSPACE_SIZE = MAX_FLOAT * MAX_FLOAT;
HACL_INLINE __aicore__ void 
hablas_store_vector_ub2gm(__gm__ float *dst, 
                           __ub__ float *src,
                           __ub__ float *wksp, 
                           int64_t valid_len,
                           int64_t incy) 
{
    if (incy == 1) {
        memcpy_ub2gm_anylen(dst, src, valid_len);
    } else {
        int64_t loop = valid_len * incy / UB_WORKSPACE_SIZE;
        int64_t remain = (valid_len * incy) % UB_WORKSPACE_SIZE;

        int64_t start_posi = 0; // 起始写入位置
        int isrc_ele = 0;
        set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        for (int idx = 0; idx < loop; ++idx) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
            _memcpy(wksp, dst + idx * UB_WORKSPACE_SIZE, UB_WORKSPACE_SIZE); // gm2ub_原生实现不需要同步
            set_flag(PIPE_MTE2, PIPE_S, 0);
            wait_flag(PIPE_MTE2, PIPE_S, 0);

            int iwhile = start_posi;
            while (iwhile < UB_WORKSPACE_SIZE) {
                *(wksp + iwhile) = *(src + isrc_ele);
                iwhile = iwhile + incy;
                isrc_ele = isrc_ele + 1;
            }
            start_posi = iwhile - UB_WORKSPACE_SIZE; 
            set_flag(PIPE_S, PIPE_MTE3, 0);
            wait_flag(PIPE_S, PIPE_MTE3, 0);
            _memcpy(dst + idx * UB_WORKSPACE_SIZE, wksp, UB_WORKSPACE_SIZE);
            set_flag(PIPE_MTE3, PIPE_MTE2, 0);
        }
        if (remain) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
            _memcpy(wksp, dst + loop * UB_WORKSPACE_SIZE, remain);
            set_flag(PIPE_MTE2, PIPE_S, 0);
            wait_flag(PIPE_MTE2, PIPE_S, 0);

            int iwhile = start_posi;
            while (isrc_ele < valid_len && iwhile < UB_WORKSPACE_SIZE) {
                *(wksp + iwhile) = *(src + isrc_ele);
                iwhile = iwhile + incy;
                isrc_ele = isrc_ele + 1;
            }
            set_flag(PIPE_S, PIPE_MTE3, 0);
            wait_flag(PIPE_S, PIPE_MTE3, 0);
            memcpy_ub2gm_anylen(dst + loop * UB_WORKSPACE_SIZE, wksp, remain);
            set_flag(PIPE_MTE3, PIPE_MTE2, 0);

        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
    }
}


extern "C" __global__ __aicore__ void hablas_ssymv_kernel(
    int64_t uplo,
    const int64_t N,
    float alpha,
    __gm__ float *A,
    const int64_t lda,
    __gm__ float *X,
    const int64_t incx,
    float beta,
    __gm__ float *Y,
    const int64_t incy,
    const int64_t kernel_N)
{

    // 列优先
    int64_t n_part = N / kernel_N;
    int64_t n_remain = N % kernel_N;
    int64_t row_cnt = n_remain == 0 ? n_part : n_part + 1;

    int64_t kernel_row = row_cnt / block_num;
    int64_t kernel_remain = row_cnt % block_num;
    if (block_idx < kernel_remain)
    {
        kernel_row += 1;
    }

    set_flag(PIPE_S, PIPE_V, 0);
    wait_flag(PIPE_S, PIPE_V, 0);

    Vector<float_8, MAX_FLOAT * MAX_FLOAT, HACL_UB> block;
    Vector<float_8, MAX_FLOAT * MAX_FLOAT, HACL_L1> block_l1;
    Vector<float_8, MAX_FLOAT, HACL_UB> x1;
    Vector<float_8, MAX_FLOAT, HACL_UB> x2;
    Vector<float_8, MAX_FLOAT, HACL_L1> x_;
    __ub__ float *x;

    Vector<float_8, MAX_FLOAT, HACL_UB> y;
    Vector<float_8, 1, HACL_UB> buffer;

    Vector<float_8, MAX_FLOAT, HACL_UB> wksp;

    set_flag(PIPE_MTE3, PIPE_S, 1); // set(1)
    for (int64_t j = 0; j < kernel_row; ++j)
    {
        
        wait_flag(PIPE_MTE3, PIPE_S, 1); // wait(1)

        int64_t row_id;
        if (block_idx < kernel_remain)
        {
            row_id = block_idx * kernel_row + j;
        }
        else
        {
            row_id = kernel_remain * (kernel_row + 1) + (block_idx - kernel_remain) * kernel_row + j;
        }
        int64_t m = (row_id == n_part ? n_remain : kernel_N);

        set_flag(PIPE_S, PIPE_MTE2, 0);
        wait_flag(PIPE_S, PIPE_MTE2, 0);
        if (incy == 1)
        {
            _memcpy(y.get_ptr(0), Y + row_id * kernel_N * incy, m);
            set_flag(PIPE_MTE2, PIPE_V, 0);
            wait_flag(PIPE_MTE2, PIPE_V, 0);
            vec_muls(y.get_ptr(0), y.get_ptr(0), beta, m);
        }
        else
        {
             set_flag(PIPE_S, PIPE_MTE2, 1);
            for (int64_t i = 0; i < m; ++i)
            {
                wait_flag(PIPE_S, PIPE_MTE2, 1);
                _memcpy(buffer.get_ptr(0), Y + row_id * kernel_N * incy + (i * incy), 1);
                set_flag(PIPE_MTE2, PIPE_S, 0);
                wait_flag(PIPE_MTE2, PIPE_S, 0);
                *(y.get_ptr(0) + i) = *(buffer.get_ptr(0)) * beta;
                set_flag(PIPE_S, PIPE_MTE2, 1);
            }
            wait_flag(PIPE_S, PIPE_MTE2, 1);
        }

        set_flag(PIPE_V, PIPE_S, 1); // set(2)
        set_flag(PIPE_MTE1, PIPE_MTE2, 2); //set(6)
        for (int64_t i = 0; i < row_cnt; i++)
        {
            wait_flag(PIPE_V, PIPE_S, 1); //wait(2)

            x = (i % 2 == 0 ? x1.get_ptr(0) : x2.get_ptr(0));

            int64_t index = row_id * row_cnt + i;

            bool tag = false;
            int64_t column_id = i;
            int64_t n = (column_id == n_part) ? n_remain : kernel_N;

            int64_t row_id_part = row_id;
            int64_t column_id_part = column_id;
            int ac_m = m;
            int ac_n = n;
            if (column_id > row_id && uplo == 0)
            {
                row_id_part = column_id;
                column_id_part = row_id;
                tag = true;
                ac_m = n;
                ac_n = m;
            }
            if (column_id < row_id && uplo == 1)
            {
                row_id_part = column_id;
                column_id_part = row_id;
                tag = true;
                ac_m = n;
                ac_n = m;
            }

            x1 = 0.0;   
            x2 = 0.0;

            set_flag(PIPE_S, PIPE_MTE2, 0);
            wait_flag(PIPE_S, PIPE_MTE2, 0);

            wait_flag(PIPE_MTE1, PIPE_MTE2, 2); //wait(6)
            load_x_L1(x_.get_ptr(0), X + column_id * kernel_N * incx, n, incx, alpha);
            set_flag(PIPE_MTE2, PIPE_MTE1, 0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, 0);
            load_x_L12UB(x, x_.get_ptr(0), n, incx, alpha);

            set_flag(PIPE_MTE1, PIPE_MTE2, 0);
            wait_flag(PIPE_MTE1, PIPE_MTE2, 0);

            load_block_gm2l1(block_l1.get_ptr(0), A + row_id_part * kernel_N + (column_id_part * (lda * kernel_N)), lda, kernel_N, ac_m, ac_n);
            set_flag(PIPE_MTE2, PIPE_V, 0);
            wait_flag(PIPE_MTE2, PIPE_V, 0);
            vec_dup(block.get_ptr(0), float(0), MAX_FLOAT * MAX_FLOAT);
            set_flag(PIPE_V, PIPE_MTE1, 0);
            wait_flag(PIPE_V, PIPE_MTE1, 0);
            load_block_l12ub(block.get_ptr(0), block_l1.get_ptr(0), lda, kernel_N, ac_m, ac_n);
            set_flag(PIPE_MTE1, PIPE_MTE2, 2); //set(6)


            set_flag(PIPE_MTE1, PIPE_S, 0);
            wait_flag(PIPE_MTE1, PIPE_S, 0);

            if (row_id_part == column_id_part)
            {
                fillup(block.get_ptr(0), m, n, uplo, kernel_N);
            }

            set_flag(PIPE_S, PIPE_V, 0);
            wait_flag(PIPE_S, PIPE_V, 0);

            if (tag)
            {
                // 转置计算
                compute_transpose(block.get_ptr(0), x, y.get_ptr(0), ac_m, ac_n, kernel_N);
            }
            else
            {
                // 非转置计算
                compute_without_transpose(block.get_ptr(0), x, y.get_ptr(0), ac_m, ac_n, kernel_N);
            }

            set_flag(PIPE_V, PIPE_S, 1); // set(2)
        }
        wait_flag(PIPE_V, PIPE_S, 1); // wait(2)
        wait_flag(PIPE_MTE1, PIPE_MTE2, 2); //wait(6)


        set_flag(PIPE_V, PIPE_MTE3, 0);
        wait_flag(PIPE_V, PIPE_MTE3, 0);
        
        hablas_store_vector_ub2gm(Y + row_id * kernel_N * incy,
                                  y.get_ptr(0),
                                  wksp.get_ptr(0),
                                  m,
                                  incy);
        set_flag(PIPE_MTE3, PIPE_S, 1); // set(1)

    }
    wait_flag(PIPE_MTE3, PIPE_S, 1); //wait(1)

}
