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
extern char hablas_htrmv_kernel;
extern char hablas_htrmv_copy_kernel;
extern char hablas_strmv_kernel;
extern char hablas_strmv_copy_kernel;
extern char hablas_ctrmv_kernel;
extern char hablas_ctrmv_copy_kernel;
extern char hablas_csymv_kernel;
extern char hablas_cgemv_kernel;
extern char hablas_htrmm_kernel;

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
    uint64_t blockDim = M < 16 ? 1 : ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
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
    uint64_t blockDim = M < 16 ? 1 : batch_count * ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
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
    uint64_t blockDim = M < 16 ? 1 : batch_count * ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
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
    uint64_t blockDim = M < 16 ? 1 : CORENUM;
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
    uint64_t blockDim = M < 16 ? 1 : CORENUM;
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
                      float alpha,
                      void *input1_hbm,
                      int64_t lda,
                      void *input2_hbm,
                      int64_t incx,
                      float beta,
                      void *input3_hbm,
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
        void *input1_hbm;
        int64_t lda;
        void *input2_hbm;
        int64_t incx;
        float beta;
        void *input3_hbm;
        int64_t incy;
    };

    KernelArgs args;
    args.trans = trans;
    args.M = M;
    args.N = N;
    args.alpha = alpha;
    args.input1_hbm = input1_hbm;
    args.lda = lda;
    args.incx = incx;
    args.input2_hbm = input2_hbm;
    args.beta = beta;
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

rtError_t hablasCsymv(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      int64_t N,
                      void *alpha,
                      void *A,
                      int64_t lda,
                      void *X,
                      int64_t incx,
                      void *beta,
                      void *Y,
                      int64_t incy)
{
    rtStream_t stream;
    rtError_t error;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_csymv_kernel";
    uint64_t blockDim = CORENUM;
    error = registerKernel(hablas_csymv_kernel, func_name);
    struct KernelArgs
    {
        int64_t uplo;
        int64_t N;
        void *alpha;
        void *A;
        int64_t lda;
        void *X;
        int64_t incx;
        void *beta;
        void *Y;
        int64_t incy;
        void *workspace;
        int64_t base_block_size;
    };

    void *workspace = nullptr;
    error = rtMalloc((void **)&workspace, (int64_t)N * sizeof(float) * 2, RT_MEMORY_HBM);

    KernelArgs args;
    if (uplo == HABLAS_FILL_MODE_LOWER)
    {
        args.uplo = 0;
    }
    else
    {
        args.uplo = 1;
    }
    args.N = N;
    args.alpha = alpha;
    args.A = A;
    args.lda = lda;
    args.incx = incx;
    args.X = X;
    args.beta = beta;
    args.Y = Y;
    args.incy = incy;
    args.workspace = workspace;
    int base_block_size = 96;
    while (N % base_block_size < 8 && N % base_block_size > 0 && base_block_size >= 16)
    {
        base_block_size -= 8;
    }
    args.base_block_size = base_block_size;
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
    error = rtStreamSynchronize(stream);
    return error;
}

rtError_t hablasHtrmv(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      hablasOperation_t transA,
                      hablasDiagType_t diag,
                      int64_t M,
                      void *A,
                      int64_t lda,
                      void *X,
                      int64_t incx)
{
    rtStream_t stream;
    rtError_t error;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_htrmv_kernel";
    uint64_t blockDim = CORENUM;
    error = registerKernel(hablas_htrmv_kernel, func_name);
    struct KernelArgs
    {
        int64_t uplo;
        int64_t transA;
        int64_t diag;
        int64_t dim_M;
        void *matrixA;
        int64_t lda;
        void *x;
        int64_t incx;
        void *workspace;
        int64_t base_block_size;
    };
    void *workspace = nullptr;
    error = rtMalloc((void **)&workspace, (int64_t)M * sizeof(__fp16) + 16, RT_MEMORY_HBM);

    KernelArgs args;
    if (uplo == HABLAS_FILL_MODE_LOWER)
    {
        args.uplo = 0;
    }
    else if (uplo == HABLAS_FILL_MODE_UPPER)
    {
        args.uplo = 1;
    }
    else
    {
        args.uplo = 2;
    }
    if (transA == HABLAS_OP_N)
    {
        args.transA = 0;
    }
    else
    {
        args.transA = 1;
    }
    if (diag == HABLAS_DIAG_NON_UNIT)
    {
        args.diag = 0;
    }
    else
    {
        args.diag = 1;
    }
    args.dim_M = M;
    args.matrixA = A;
    args.lda = lda;
    args.x = X;
    args.incx = incx;
    args.workspace = workspace;

    int base_block_size = 128;
    while (M % base_block_size < 16 && M % base_block_size > 0 && base_block_size > 16)
    {
        base_block_size -= 16;
    }
    args.base_block_size = base_block_size;
    std::cout << "BLOCK SIZE:" << base_block_size << std::endl;

    error = rtKernelLaunch(func_name,
                           blockDim,
                           (void *)(&args),
                           sizeof(args),
                           NULL,
                           stream);
    error = rtStreamSynchronize(stream);

    const char *func_name1 = "hablas_htrmv_copy_kernel";
    error = registerKernel(hablas_htrmv_copy_kernel, func_name1);
    struct KernelArgs1
    {
        int64_t dim_M;
        void *x;
        int64_t incx;
        void *workspace;
    };
    blockDim = 1;
    KernelArgs1 args1;
    args1.dim_M = M;
    args1.incx = incx;
    args1.x = X;
    args1.workspace = workspace;
    error = rtKernelLaunch(func_name1,
                           blockDim,
                           (void *)(&args1),
                           sizeof(args1),
                           NULL,
                           stream);
    error = rtStreamSynchronize(stream);
    return error;
}

rtError_t hablasCtrmv(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      hablasOperation_t transA,
                      hablasDiagType_t diag,
                      int64_t M,
                      void *A,
                      int64_t lda,
                      void *X,
                      int64_t incx)
{
    rtStream_t stream;
    rtError_t error;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_ctrmv_kernel";
    uint64_t blockDim = CORENUM;
    error = registerKernel(hablas_ctrmv_kernel, func_name);
    struct KernelArgs
    {
        int64_t uplo;
        int64_t transA;
        int64_t diag;
        int64_t dim_M;
        void *matrixA;
        int64_t lda;
        void *x;
        int64_t incx;
        void *workspace;
        void *workspace1;
        int64_t base_block_size;
    };
    void *workspace = nullptr;
    error = rtMalloc((void **)&workspace, (int64_t)M * sizeof(haComplex) + 16, RT_MEMORY_HBM);
    void *workspace1 = nullptr;
    error = rtMalloc((void **)&workspace1, (int64_t)M * sizeof(haComplex) + 16, RT_MEMORY_HBM);

    KernelArgs args;
    if (uplo == HABLAS_FILL_MODE_LOWER)
    {
        args.uplo = 0;
    }
    else if (uplo == HABLAS_FILL_MODE_UPPER)
    {
        args.uplo = 1;
    }
    else
    {
        args.uplo = 2;
    }
    if (transA == HABLAS_OP_N)
    {
        args.transA = 0;
    }
    else if (transA == HABLAS_OP_T)
    {
        args.transA = 1;
    }
    else
    {
        args.transA = 2;
    }
    if (diag == HABLAS_DIAG_NON_UNIT)
    {
        args.diag = 0;
    }
    else
    {
        args.diag = 1;
    }
    args.dim_M = M;
    args.matrixA = A;
    args.lda = lda;
    args.x = X;
    args.incx = incx;
    args.workspace = workspace;
    args.workspace1 = workspace1;

    int base_block_size = 80;
    while (M % base_block_size < 8 && M % base_block_size > 0 && base_block_size > 16)
    {
        base_block_size -= 8;
    }
    args.base_block_size = base_block_size;

    error = rtKernelLaunch(func_name,
                           blockDim,
                           (void *)(&args),
                           sizeof(args),
                           NULL,
                           stream);
    error = rtStreamSynchronize(stream);

    const char *func_name1 = "hablas_ctrmv_copy_kernel";
    error = registerKernel(hablas_ctrmv_copy_kernel, func_name1);
    struct KernelArgs1
    {
        int64_t dim_M;
        void *x;
        int64_t incx;
        void *workspace;
    };
    blockDim = 1;
    KernelArgs1 args1;
    args1.dim_M = M;
    args1.incx = incx;
    args1.x = X;
    args1.workspace = workspace;
    error = rtKernelLaunch(func_name1,
                           blockDim,
                           (void *)(&args1),
                           sizeof(args1),
                           NULL,
                           stream);
    error = rtStreamSynchronize(stream);
    return error;
}

rtError_t hablasStrmv(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      hablasOperation_t transA,
                      hablasDiagType_t diag,
                      int64_t M,
                      void *A,
                      int64_t lda,
                      void *X,
                      int64_t incx)
{
    rtStream_t stream;
    rtError_t error;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_strmv_kernel";
    uint64_t blockDim = CORENUM;
    error = registerKernel(hablas_strmv_kernel, func_name);
    struct KernelArgs
    {
        int64_t uplo;
        int64_t transA;
        int64_t diag;
        int64_t dim_M;
        void *matrixA;
        int64_t lda;
        void *x;
        int64_t incx;
        void *workspace;
        int64_t base_block_size;
    };
    void *workspace = nullptr;
    error = rtMalloc((void **)&workspace, (int64_t)M * sizeof(float) + 8, RT_MEMORY_HBM);

    KernelArgs args;
    if (uplo == HABLAS_FILL_MODE_LOWER)
    {
        args.uplo = 0;
    }
    else if (uplo == HABLAS_FILL_MODE_UPPER)
    {
        args.uplo = 1;
    }
    else
    {
        args.uplo = 2;
    }
    if (transA == HABLAS_OP_N)
    {
        args.transA = 0;
    }
    else
    {
        args.transA = 1;
    }
    if (diag == HABLAS_DIAG_NON_UNIT)
    {
        args.diag = 0;
    }
    else
    {
        args.diag = 1;
    }
    args.dim_M = M;
    args.matrixA = A;
    args.lda = lda;
    args.x = X;
    args.incx = incx;
    args.workspace = workspace;

    int base_block_size = 128;
    while (M % base_block_size < 8 && M % base_block_size > 0 && base_block_size > 16)
    {
        base_block_size -= 8;
    }
    args.base_block_size = base_block_size;

    error = rtKernelLaunch(func_name,
                           blockDim,
                           (void *)(&args),
                           sizeof(args),
                           NULL,
                           stream);
    error = rtStreamSynchronize(stream);

    const char *func_name1 = "hablas_strmv_copy_kernel";
    error = registerKernel(hablas_strmv_copy_kernel, func_name1);
    struct KernelArgs1
    {
        int64_t dim_M;
        void *x;
        int64_t incx;
        void *workspace;
    };
    blockDim = 1;
    KernelArgs1 args1;
    args1.dim_M = M;
    args1.incx = incx;
    args1.x = X;
    args1.workspace = workspace;
    error = rtKernelLaunch(func_name1,
                           blockDim,
                           (void *)(&args1),
                           sizeof(args1),
                           NULL,
                           stream);
    error = rtStreamSynchronize(stream);
    return error;
}

rtError_t hablasHtrmm(hablasHandle_t handle,
                      hablasSideMode_t side,
                      hablasFillMode_t uplo,
                      hablasOperation_t transA,
                      hablasDiagType_t diag,
                      int64_t M,
                      int64_t N,
                      __fp16 *alpha,
                      __fp16 *A_d,
                      int64_t lda,
                      __fp16 *B_d,
                      int64_t ldb,
                      __fp16 *C_d,
                      int64_t ldc)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_htrmm_kernel";
    uint64_t blockDim = M < 16 ? 1 : ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
    std::cout << "blockDim: " << blockDim << std::endl;
    error = registerKernel(hablas_htrmm_kernel, func_name);

    int64_t A_SIZE;
    int64_t B_SIZE = ldb * N;
    if (side == HABLAS_SIDE_LEFT)
    {
        A_SIZE = lda * M;
    }
    else
    {
        lda = N;
        A_SIZE = lda * N;
    }

    struct KernelArgs
    {
        hablasSideMode_t side;
        hablasFillMode_t uplo;
        hablasOperation_t transA;
        hablasDiagType_t diag;
        int64_t M;
        int64_t N;
        __fp16 alpha;
        void *matrixA;
        int64_t lda;
        void *matrixB;
        int64_t ldb;
        void *matrixC;
        int64_t ldc;
    };
    KernelArgs args;
    args.side = side;
    args.uplo = uplo;
    args.transA = transA;
    args.diag = diag;
    args.M = M;
    args.N = N;
    args.alpha = *alpha;
    args.matrixA = A_d;
    args.lda = lda;
    args.matrixB = B_d;
    args.ldb = ldb;
    args.matrixC = C_d;
    args.ldc = ldc;

    error = rtKernelLaunch((void *)func_name,
                           blockDim,
                           (void *)(&args),
                           sizeof(args),
                           NULL,
                           stream);

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
