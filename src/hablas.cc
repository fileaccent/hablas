#include "hablas.h"
#include <iostream>

extern char hablas_hgemm_kernel;
extern char hablas_hgemm_batched_kernel;
extern char hablas_hgemm_strided_batched_kernel;
extern char hablas_hsyrk_kernel;
extern char hablas_hsyr2k_kernel;
extern char hablas_hgemv_kernel;
extern char hablas_sgemv_kernel;
extern char hablas_ssymv_kernel;
extern char hablas_hsymv_kernel;

rtError_t registerKernel(char &k, const char *func_name)
{
    char *p = &k;
    int len = *((int *)p);
    char *start = p + 64;
    char *end = p + 64 + len;
    rtDevBinary_t binary;
    void *binHandle = NULL;
    uint32_t bufferSize = len;

    binary.data = start;
    binary.length = len;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;
    rtError_t rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != RT_ERROR_NONE)
    {
        printf("[FAILED]rtDevBinaryRegister failed!\n");
    }
    else
    {
        printf("[SUCCESS]rtDevBinaryRegister succeed!\n");
    }
    rtRet = rtFunctionRegister(binHandle, func_name, func_name, (void *)func_name, 0);
    if (rtRet != RT_ERROR_NONE)
    {
        printf("[FAILED]rtFunctionRegister failed!\n");
    }
    else
    {
        printf("[SUCCESS]rtFunctionRegister succeed!\n");
    }
    return rtRet;
}

rtError_t hablasHgemm(hablasHandle_t handle,
                      hablasOperation_t transA,
                      hablasOperation_t transB,
                      int64_t M,
                      int64_t N,
                      int64_t K,
                      __fp16 *alpha,
                      __fp16 *A_d,
                      int64_t lda,
                      __fp16 *B_d,
                      int64_t ldb,
                      __fp16 *beta,
                      __fp16 *C_d,
                      int64_t ldc)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hgemm_kernel";
    uint64_t blockDim = ldc < 16 ? 1 : ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
    std::cout << "blockDim: " << blockDim << std::endl;
    error = registerKernel(hablas_hgemm_kernel, func_name);
    struct KernelArgs
    {
        hablasOperation_t transA;
        hablasOperation_t transB;
        int64_t M;
        int64_t N;
        int64_t K;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        void *matrixB;
        int64_t ldb;
        __fp16 beta;
        void *matrixC;
        int64_t ldc;
    };
    KernelArgs args;
    args.transA = transA;
    args.transB = transB;
    args.M = M;
    args.N = N;
    args.K = K;
    args.alpha = *alpha;
    args.matrixA = A_d;
    args.lda = lda;
    args.matrixB = B_d;
    args.ldb = ldb;
    args.beta = *beta;
    args.matrixC = C_d;
    args.ldc = ldc;

    error = rtKernelLaunch((void *)func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);

    if (error == RT_ERROR_NONE)
    {
        printf("[SUCCESS]rtKernelLaunch succeed!\n");
    }
    else
    {
        printf("[FAILED]rtKernelLaunch failed!\n");
    }
    return error;
}

rtError_t hablasHgemmBatched(hablasHandle_t handle,
                             hablasOperation_t transA,
                             hablasOperation_t transB,
                             int64_t M,
                             int64_t N,
                             int64_t K,
                             __fp16 *alpha,
                             void *matrixA,
                             int64_t lda,
                             void *matrixB,
                             int64_t ldb,
                             __fp16 *beta,
                             void *matrixC,
                             int64_t ldc,
                             int64_t batch_count)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hgemm_batched_kernel";
    uint64_t blockDim = ldc < 16 ? 1 : batch_count * ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
    error = registerKernel(hablas_hgemm_batched_kernel, func_name);

    struct KernelArgs
    {
        hablasOperation_t transA;
        hablasOperation_t transB;
        int64_t M;
        int64_t N;
        int64_t K;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        void *matrixB;
        int64_t ldb;
        __fp16 beta;
        void *matrixC;
        int64_t ldc;
        int64_t batch_count;
    };

    KernelArgs args;
    __fp16 *workspace = NULL;
    int64_t *matrixC_pad;
    int64_t *matrixC_pad_h;
    args.transA = transA;
    args.transB = transB;
    args.M = M;
    args.N = N;
    args.K = K;
    args.alpha = *alpha;
    args.matrixA = matrixA;
    args.lda = lda;
    args.matrixB = matrixB;
    args.ldb = ldb;
    args.beta = *beta;
    args.matrixC = matrixC;
    args.ldc = ldc;
    args.batch_count = batch_count;

    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    if (error == RT_ERROR_NONE)
    {
        printf("[SUCCESS]rtKernelLaunch succeed!\n");
    }
    else
    {
        printf("[FAILED]rtKernelLaunch failed!\n");
    }
    return error;
}

rtError_t hablasHgemmStridedBatched(hablasHandle_t handle,
                                    hablasOperation_t transA,
                                    hablasOperation_t transB,
                                    int64_t M,
                                    int64_t N,
                                    int64_t K,
                                    __fp16 *alpha,
                                    __fp16 *const matrixA,
                                    int64_t lda,
                                    int64_t strideA,
                                    __fp16 *const matrixB,
                                    int64_t ldb,
                                    int64_t strideB,
                                    __fp16 *beta,
                                    __fp16 *matrixC,
                                    int64_t ldc,
                                    int64_t strideC,
                                    int64_t batch_count)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hgemm_strided_batched_kernel";
    error = registerKernel(hablas_hgemm_strided_batched_kernel, func_name);
    uint64_t blockDim = batch_count * ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
    struct KernelArgs
    {
        hablasOperation_t transA;
        hablasOperation_t transB;
        int64_t M;
        int64_t N;
        int64_t K;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        int64_t strideA;
        void *matrixB;
        int64_t ldb;
        int64_t strideB;
        __fp16 beta;
        void *matrixC;
        int64_t ldc;
        int64_t strideC;
        int64_t batch_count;
    };
    KernelArgs args;
    args.transA = transA;
    args.transB = transB;
    args.M = M;
    args.N = N;
    args.K = K;
    args.alpha = *alpha;
    args.matrixA = matrixA;
    args.lda = lda;
    args.strideA = strideA;
    args.matrixB = matrixB;
    args.ldb = ldb;
    args.strideB = strideB;
    args.beta = *beta;
    args.matrixC = matrixC;
    args.ldc = ldc;
    args.strideC = strideC;
    args.batch_count = batch_count;

    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    if (error == RT_ERROR_NONE)
    {
        printf("[SUCCESS]rtKernelLaunch succeed!\n");
    }
    else
    {
        printf("[FAILED]rtKernelLaunch failed!\n");
    }
    return error;
}

rtError_t hablasHsyrk(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      hablasOperation_t transA,
                      int64_t N,
                      int64_t K,
                      __fp16 *alpha,
                      __fp16 *matrixA,
                      int64_t lda,
                      __fp16 *beta,
                      __fp16 *matrixC,
                      int64_t ldc)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hsyrk_kernel";
    uint64_t blockDim = ldc < 16 ? 1 : CORENUM;
    error = registerKernel(hablas_hsyrk_kernel, func_name);
    struct KernelArgs
    {
        hablasFillMode_t uplo;
        hablasOperation_t transA;
        int64_t N;
        int64_t K;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        __fp16 beta;
        void *matrixC;
        int64_t ldc;
    };

    KernelArgs args;
    args.uplo = uplo;
    args.transA = transA;
    args.N = N;
    args.K = K;
    args.alpha = *alpha;
    args.matrixA = matrixA;
    args.lda = lda;
    args.beta = *beta;
    args.matrixC = matrixC;
    args.ldc = ldc;
    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    if (error == RT_ERROR_NONE)
    {
        printf("[SUCCESS]rtKernelLaunch succeed!\n");
    }
    else
    {
        printf("[FAILED]rtKernelLaunch failed!\n");
    }
    return error;
}

rtError_t hablasHsyr2k(hablasHandle_t handle,
                       hablasFillMode_t uplo,
                       hablasOperation_t trans,
                       int64_t N,
                       int64_t K,
                       __fp16 *alpha,
                       __fp16 *matrixA,
                       int64_t lda,
                       __fp16 *matrixB,
                       int64_t ldb,
                       __fp16 *beta,
                       __fp16 *matrixC,
                       int64_t ldc)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hsyr2k_kernel";
    uint64_t blockDim = ldc < 16 ? 1 : CORENUM;
    error = registerKernel(hablas_hsyr2k_kernel, func_name);

    struct KernelArgs
    {
        hablasFillMode_t uplo;
        hablasOperation_t transA;
        int64_t N;
        int64_t K;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        void *matrixB;
        int64_t ldb;
        __fp16 beta;
        void *matrixC;
        int64_t ldc;
    };

    KernelArgs args;
    args.uplo = uplo;
    args.transA = trans;
    args.N = N;
    args.K = K;
    args.alpha = *alpha;
    args.matrixA = matrixA;
    args.lda = lda;
    args.matrixB = matrixB;
    args.ldb = ldb;
    args.beta = *beta;
    args.matrixC = matrixC;
    args.ldc = ldc;

    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    if (error == RT_ERROR_NONE)
    {
        printf("[SUCCESS]rtKernelLaunch succeed!\n");
    }
    else
    {
        printf("[FAILED]rtKernelLaunch failed!\n");
    }
    return error;
}

rtError_t hablasHgemv(hablasHandle_t handle,
                      hablasOperation_t trans,
                      int64_t M,
                      int64_t N,
                      __fp16 *alpha,
                      __fp16 *h_A,
                      int64_t lda,
                      __fp16 *h_X,
                      int64_t incx,
                      __fp16 *beta,
                      __fp16 *h_Y,
                      int64_t incy)
{
    rtError_t error;
    if (trans == HABLAS_OP_N && incx == 1 && incy == 1)
    {
        error = hablasHgemm(handle, trans, HABLAS_OP_N, M, 1, N, alpha, h_A, lda, h_X, N, beta, h_Y, M);
        return error;
    }
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hgemv_kernel";
    uint64_t blockDim = (M + 64 - 1) / 64;
    if (trans == HABLAS_OP_T)
    {
        blockDim = (N + 64 - 1) / 64;
    }
    error = registerKernel(hablas_hgemv_kernel, func_name);
    struct KernelArgs
    {
        hablasOperation_t trans;
        int64_t M;
        int64_t N;
        __fp16 alpha;
        __fp16 *h_A;
        int64_t lda;
        __fp16 *h_X;
        int64_t incx;
        __fp16 beta;
        __fp16 *h_Y;
        int64_t incy;
    };

    KernelArgs args;
    args.trans = trans;
    args.M = M;
    args.N = N;
    args.alpha = *alpha;
    args.h_A = h_A;
    args.lda = lda;
    args.h_X = h_X;
    args.incx = incx;
    args.beta = *beta;
    args.h_Y = h_Y;
    args.incy = incy;
    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    return error;
}

rtError_t hablasSgemv(hablasHandle_t handle,
                      hablasOperation_t trans,
                      int64_t M,
                      int64_t N,
                      float *alpha,
                      float *input1_hbm,
                      int64_t lda,
                      float *input2_hbm,
                      int64_t incx,
                      float *beta,
                      float *input3_hbm,
                      int64_t incy)
{
    rtStream_t stream;
    rtError_t error;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_sgemv_kernel";
    uint64_t blockDim = (M + 512 - 1) / 512;
    if (trans == HABLAS_OP_T)
    {
        blockDim = (N + 64 - 1) / 64;
    }
    error = registerKernel(hablas_sgemv_kernel, func_name);
    struct KernelArgs
    {
        hablasOperation_t trans;
        int64_t M;
        int64_t N;
        float alpha;
        float *input1_hbm;
        int64_t lda;
        float *input2_hbm;
        int64_t incx;
        float beta;
        float *input3_hbm;
        int64_t incy;
    };

    KernelArgs args;
    args.trans = trans;
    args.M = M;
    args.N = N;
    args.alpha = *alpha;
    args.input1_hbm = input1_hbm;
    args.lda = lda;
    args.incx = incx;
    args.input2_hbm = input2_hbm;
    args.beta = *beta;
    args.input3_hbm = input3_hbm;
    args.incy = incy;
    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    return error;
}

rtError_t hablasHsymv(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      int64_t N,
                      __fp16 alpha,
                      __fp16 *A,
                      int64_t lda,
                      __fp16 *x,
                      int64_t incx,
                      __fp16 beta,
                      __fp16 *y,
                      int64_t incy)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hsymv_kernel";
    uint64_t blockDim = CORENUM;
    error = registerKernel(hablas_hsymv_kernel, func_name);

    struct KernelArgs
    {
        hablasFillMode_t uplo;
        int64_t n;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        void *x;
        int64_t incx;
        __fp16 beta;
        void *y;
        int64_t incy;
    };
    KernelArgs args;

    args.uplo = uplo;
    args.n = N;
    args.alpha = alpha;
    args.matrixA = A;
    args.lda = lda;
    args.x = x;
    args.incx = incx;
    args.beta = beta;
    args.y = y;
    args.incy = incy;

    error = rtKernelLaunch("hablas_hsymv_kernel", blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    if (error == RT_ERROR_NONE)
    {
        printf("[SUCCESS]rtKernelLaunch succeed!\n");
    }
    else
    {
        printf("[FAILED]rtKernelLaunch failed!\n");
    }
    return error;
}

rtError_t hablasSsymv(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      int64_t N,
                      float alpha,
                      void *A,
                      int64_t lda,
                      void *X,
                      int64_t incx,
                      float beta,
                      void *Y,
                      int64_t incy)
{
    rtStream_t stream;
    rtError_t error;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_ssymv_kernel";
    uint64_t blockDim = CORENUM;
    error = registerKernel(hablas_ssymv_kernel, func_name);
    struct KernelArgs
    {
        hablasFillMode_t uplo;
        int64_t N;
        float alpha;
        void *A;
        int64_t lda;
        void *X;
        int64_t incx;
        float beta;
        void *Y;
        int64_t incy;
        int64_t Kernel_N;
    };
    KernelArgs args;
    args.uplo = uplo;
    args.N = N;
    args.alpha = alpha;
    args.A = A;
    args.lda = lda;
    args.incx = incx;
    args.X = X;
    args.beta = beta;
    args.Y = Y;
    args.incy = incy;
    args.Kernel_N = 128;
    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    return error;
}