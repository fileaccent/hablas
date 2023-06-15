#include <hacl/hacl.h>

extern "C"   __global__ __aicore__
void hablas_htrmv_copy_kernel(int64_t M,
                              __gm__ half *vectorX,
                              int64_t incx,
                              __gm__ half *workspace)
{
    Vector<half_16, 128 * 128 / 16, HACL_UB> ub_tmpw;
    Vector<half_16, 128 * 128 / 16, HACL_UB> ub_tmpx;

    int64_t cont_tmpw = 128 * 128;
    int64_t cont_tmpx = 128 * 128;
    
    int64_t loop_tmpw = M / cont_tmpw;
    int64_t remain_tmpw = M % cont_tmpw;

    if (incx == 1) {
        for (int64_t w_idx = 0; w_idx < loop_tmpw; ++ w_idx) {
            pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + w_idx * cont_tmpw, cont_tmpw);
            pipe_barrier(PIPE_ALL);
            _memcpy(vectorX + w_idx * cont_tmpw, ub_tmpw.get_ptr(0), cont_tmpw);
        }
        if (remain_tmpw) {
            pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + loop_tmpw * cont_tmpw, remain_tmpw);
            pipe_barrier(PIPE_ALL);
            if (remain_tmpw < 16) {
                __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(vectorX + loop_tmpw * cont_tmpw, ub_tmpw.get_ptr(0), 1, 1, 0, 0);
            } else {
                _memcpy(vectorX + loop_tmpw * cont_tmpw, ub_tmpw.get_ptr(0), remain_tmpw);
            }
        }
    } else {
        for (int64_t w_idx = 0; w_idx < loop_tmpw; ++w_idx) {
            pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + w_idx * cont_tmpw, cont_tmpw);

            int64_t loop_tmpx =  (cont_tmpw * incx) / cont_tmpx;
            int64_t remain_tmpx = (cont_tmpw * incx) % cont_tmpx;
            int64_t start_posi = 0; // 起始写入位置
            int64_t iub_tmpw = 0; // 起始写入位置
            for (int64_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
                pipe_barrier(PIPE_ALL);
                _memcpy(ub_tmpx.get_ptr(0), vectorX + w_idx * cont_tmpw * incx + x_idx * cont_tmpx, cont_tmpx);

                pipe_barrier(PIPE_ALL);
                int iwhile = start_posi;
                while (iwhile < cont_tmpx) {
                    *(ub_tmpx.get_ptr(0) + iwhile) = *(ub_tmpw.get_ptr(0) + iub_tmpw);
                    iwhile = iwhile + incx;
                    iub_tmpw = iub_tmpw + 1;
                }
                start_posi = iwhile - cont_tmpx; 

                pipe_barrier(PIPE_ALL);
                _memcpy(vectorX + w_idx * cont_tmpw * incx + x_idx * cont_tmpx, ub_tmpx.get_ptr(0), cont_tmpx);
            }
            if (remain_tmpx) {
                pipe_barrier(PIPE_ALL);
                _memcpy(ub_tmpx.get_ptr(0), vectorX + w_idx * cont_tmpw * incx + loop_tmpx * cont_tmpx, remain_tmpx);
                int iwhile = start_posi;
                while (iub_tmpw < cont_tmpw && iwhile < cont_tmpx) {
                    *(ub_tmpx.get_ptr(0) + iwhile) = *(ub_tmpw.get_ptr(0) + iub_tmpw);
                    iwhile = iwhile + incx;
                    iub_tmpw = iub_tmpw + 1;
                }
                // remain_tmpx需要大于16 否则写入错误
                pipe_barrier(PIPE_ALL);
                if (remain_tmpx < 16) { 
                    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(vectorX + w_idx * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), 1, 1, 0, 0);
                } else {
                    _memcpy(vectorX + w_idx * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), remain_tmpx);
                }
            }
        }

        if (remain_tmpw) {
            pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + loop_tmpw * cont_tmpw, remain_tmpw);

            int64_t loop_tmpx =  (remain_tmpw * incx) / cont_tmpx;
            int64_t remain_tmpx = (remain_tmpw * incx) % cont_tmpx;
            int64_t start_posi = 0; // 起始写入位置
            int64_t iub_tmpw = 0;
            for (int64_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
                pipe_barrier(PIPE_ALL);
                _memcpy(ub_tmpx.get_ptr(0), vectorX + loop_tmpw * cont_tmpw * incx + x_idx * cont_tmpx, cont_tmpx);

                pipe_barrier(PIPE_ALL);
                int iwhile = start_posi;
                while (iwhile < cont_tmpx) {
                    *(ub_tmpx.get_ptr(0) + iwhile) = *(ub_tmpw.get_ptr(0) + iub_tmpw);
                    iwhile = iwhile + incx;
                    iub_tmpw = iub_tmpw + 1;
                }
                start_posi = iwhile - cont_tmpx; 

                pipe_barrier(PIPE_ALL);
                _memcpy(vectorX + loop_tmpw * cont_tmpw * incx + x_idx * cont_tmpx, ub_tmpx.get_ptr(0), cont_tmpx);
            }
            if (remain_tmpx) {
                pipe_barrier(PIPE_ALL);
                _memcpy(ub_tmpx.get_ptr(0), vectorX + loop_tmpw * cont_tmpw * incx + loop_tmpx * cont_tmpx, remain_tmpx);

                pipe_barrier(PIPE_ALL);
                int iwhile = start_posi;
                while (iub_tmpw < remain_tmpw && iwhile < cont_tmpx) {
                    *(ub_tmpx.get_ptr(0) + iwhile) = *(ub_tmpw.get_ptr(0) + iub_tmpw);
                    iwhile = iwhile + incx;
                    iub_tmpw = iub_tmpw + 1;
                }
                // remain_tmpx需要大于16 否则写入错误
                pipe_barrier(PIPE_ALL);
                if (remain_tmpx < 16) { 
                    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(vectorX + loop_tmpw * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), 1, 1, 0, 0);
                } else {
                    _memcpy(vectorX + loop_tmpw * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), remain_tmpx);
                }
            }
        }
    }
}