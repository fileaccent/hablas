#include <hacl/hacl.h>

extern "C"   __global__ __aicore__
void hablas_ctrmv_copy_kernel(int64_t M,
                              __gm__ float *vectorX,
                              int64_t incx,
                              __gm__ float *workspace)
{
    Vector<float_8, 128 * 128 / 8, HACL_UB> ub_tmpw;
    Vector<float_8, 128 * 128 / 8, HACL_UB> ub_tmpx;

    int32_t cont_tmpw = 128 * 128;
    int32_t cont_tmpx = 128 * 128;
    
    int32_t loop_tmpw = M * 2 / cont_tmpw;
    int32_t remain_tmpw = M * 2 % cont_tmpw;

    if (incx == 1) {
        for (int32_t w_idx = 0; w_idx < loop_tmpw; ++ w_idx) {
pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + w_idx * cont_tmpw, cont_tmpw);
pipe_barrier(PIPE_ALL);
            _memcpy(vectorX + w_idx * cont_tmpw, ub_tmpw.get_ptr(0), cont_tmpw); 
        }
        if (remain_tmpw) {
pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + loop_tmpw * cont_tmpw, remain_tmpw);
pipe_barrier(PIPE_ALL);
            if (remain_tmpw < 8) {
                __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(vectorX + loop_tmpw * cont_tmpw, ub_tmpw.get_ptr(0), 1, 1, 0, 0);
            } else {
                _memcpy(vectorX + loop_tmpw * cont_tmpw, ub_tmpw.get_ptr(0), remain_tmpw);
            }
        }
    } else {
        for (int32_t w_idx = 0; w_idx < loop_tmpw; ++w_idx) {
pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + w_idx * cont_tmpw, cont_tmpw);

            int32_t loop_tmpx =  (cont_tmpw / 2 * incx * 2) / cont_tmpx;
            int32_t remain_tmpx = (cont_tmpw / 2 * incx * 2) % cont_tmpx;
            int32_t start_posi = 0; // 起始写入位置
            int32_t iub_tmpw = 0; // 起始写入位置
            for (int32_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
pipe_barrier(PIPE_ALL);
                _memcpy(ub_tmpx.get_ptr(0), vectorX + w_idx * cont_tmpw * incx + x_idx * cont_tmpx, cont_tmpx);

pipe_barrier(PIPE_ALL);
                int iwhile = start_posi;
                while (iwhile < cont_tmpx) {
                    *(ub_tmpx.get_ptr(0) + iwhile) = *(ub_tmpw.get_ptr(0) + iub_tmpw);
                    *(ub_tmpx.get_ptr(0) + iwhile + 1) = *(ub_tmpw.get_ptr(0) + iub_tmpw + 1);
                    iwhile = iwhile + incx * 2;
                    iub_tmpw = iub_tmpw + 2;
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
                    *(ub_tmpx.get_ptr(0) + iwhile + 1) = *(ub_tmpw.get_ptr(0) + iub_tmpw + 1);
                    iwhile = iwhile + incx * 2;
                    iub_tmpw = iub_tmpw + 2;
                }
                // remain_tmpx需要大于16 否则写入错误
pipe_barrier(PIPE_ALL);
                if (remain_tmpx < 8) { 
                    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(vectorX + w_idx * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), 1, 1, 0, 0);
                } else {
                    _memcpy(vectorX + w_idx * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), remain_tmpx);
                }
            }
        }

        if (remain_tmpw) {
pipe_barrier(PIPE_ALL);
            _memcpy(ub_tmpw.get_ptr(0), workspace + loop_tmpw * cont_tmpw, remain_tmpw);

            int32_t loop_tmpx =  (remain_tmpw * incx) / cont_tmpx;
            int32_t remain_tmpx = (remain_tmpw * incx) % cont_tmpx;
            int32_t start_posi = 0; // 起始写入位置
            int32_t iub_tmpw = 0;
            for (int32_t x_idx = 0; x_idx < loop_tmpx; ++x_idx) {
pipe_barrier(PIPE_ALL);
                _memcpy(ub_tmpx.get_ptr(0), vectorX + loop_tmpw * cont_tmpw * incx + x_idx * cont_tmpx, cont_tmpx);

pipe_barrier(PIPE_ALL);
                int iwhile = start_posi;
                while (iwhile < cont_tmpx) {
                    *(ub_tmpx.get_ptr(0) + iwhile) = *(ub_tmpw.get_ptr(0) + iub_tmpw);
                    *(ub_tmpx.get_ptr(0) + iwhile + 1) = *(ub_tmpw.get_ptr(0) + iub_tmpw + 1);
                    iwhile = iwhile + incx * 2;
                    iub_tmpw = iub_tmpw + 2;
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
                    *(ub_tmpx.get_ptr(0) + iwhile + 1) = *(ub_tmpw.get_ptr(0) + iub_tmpw + 1);
                    iwhile = iwhile + incx * 2;
                    iub_tmpw = iub_tmpw + 2;
                }
                // remain_tmpx需要大于16 否则写入错误
pipe_barrier(PIPE_ALL);
                if (remain_tmpx < 8) { 
                    __hacl_details__::__hacl_intrinsic__memcpy_ub_gm(vectorX + loop_tmpw * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), 1, 1, 0, 0);
                } else {
                    _memcpy(vectorX + loop_tmpw * cont_tmpw * incx + loop_tmpx * cont_tmpx, ub_tmpx.get_ptr(0), remain_tmpx);
                }

            }
        }
    }
}