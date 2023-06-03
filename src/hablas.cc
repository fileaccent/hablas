#include "hablas.h"
#include <iostream>

extern char hablas_hgemm_kernel;
extern char hablas_hgemm_batched_kernel;
extern char hablas_hgemm_strided_batched_kernel;
extern char hablas_hsyrk_kernel;
extern char hablas_hsyr2k_kernel;

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
                      __fp16 alpha,
                      __fp16 *A_d,
                      int64_t lda,
                      __fp16 *B_d,
                      int64_t ldb,
                      __fp16 beta,
                      __fp16 *C_d,
                      int64_t ldc)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hgemm_kernel";
    uint64_t blockDim = ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);
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
    __fp16 *workspace = NULL;
    if (ldc < 16)
    {
        error = rtMalloc((void **)&workspace, N * 16 * sizeof(__fp16), RT_MEMORY_HBM);
        for (int64_t i = 0; i < N; ++i)
        {
            error = rtMemcpy(workspace + 16 * i, sizeof(__fp16) * M, C_d + ldc * i, sizeof(__fp16) * M, RT_MEMCPY_DEVICE_TO_DEVICE);
        }
        args.ldc = 16;
        args.matrixC = workspace;
    }
    else
    {
        args.ldc = ldc;
        args.matrixC = C_d;
    }
    args.transA = transA;
    args.transB = transB;
    args.M = M;
    args.N = N;
    args.K = K;
    args.alpha = alpha;
    args.matrixA = A_d;
    args.lda = lda;
    args.matrixB = B_d;
    args.ldb = ldb;
    args.beta = beta;

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
    if (ldc < 16)
    {
        for (int64_t i = 0; i < N; ++i)
        {
            error = rtMemcpy(C_d + ldc * i, sizeof(__fp16) * ldc, workspace + 16 * i, sizeof(__fp16) * ldc, RT_MEMCPY_DEVICE_TO_DEVICE);
        }
        error = rtFree(workspace);
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
    if (ldc < 16)
    {
        error = rtMalloc((void **)&workspace, batch_count * N * 16 * sizeof(__fp16), RT_MEMORY_HBM);
        error = rtMalloc((void **)&matrixC_pad, batch_count * sizeof(__fp16 *), RT_MEMORY_HBM);
        matrixC_pad_h = new int64_t[batch_count];
        error = rtMemcpy(matrixC_pad_h, sizeof(__fp16 *) * batch_count, matrixC, sizeof(__fp16 *) * batch_count, RT_MEMCPY_DEVICE_TO_HOST);
        for (int64_t i = 0; i < batch_count; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                error = rtMemcpy(workspace + N * 16 * i + 16 * j, sizeof(__fp16) * M, reinterpret_cast<__fp16 *>(matrixC_pad_h[i]) + ldc * j, sizeof(__fp16) * M, RT_MEMCPY_DEVICE_TO_DEVICE);
            }
            matrixC_pad_h[i] = reinterpret_cast<int64_t>(workspace + N * 16 * i);
        }
        error = rtMemcpy(matrixC_pad, sizeof(__fp16 *) * batch_count, matrixC_pad_h, sizeof(__fp16 *) * batch_count, RT_MEMCPY_HOST_TO_DEVICE);

        args.ldc = 16;
        args.matrixC = matrixC_pad;
    }
    else
    {
        args.matrixC = matrixC;
        args.ldc = ldc;
    }
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
    args.batch_count = batch_count;

    uint64_t blockDim = batch_count * ((M - 1) / 256 + 1) * ((N - 1) / 256 + 1);

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
    if (ldc < 16)
    {
        error = rtMemcpy(matrixC_pad_h, sizeof(__fp16 *) * batch_count, matrixC, sizeof(__fp16 *) * batch_count, RT_MEMCPY_DEVICE_TO_HOST);
        for (int i = 0; i < batch_count; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                error = rtMemcpy(reinterpret_cast<__fp16 *>(matrixC_pad_h[i]) + ldc * j, sizeof(__fp16) * ldc, workspace + N * 16 * i + 16 * j, sizeof(__fp16) * ldc, RT_MEMCPY_DEVICE_TO_DEVICE);
            }
        }
        delete matrixC_pad_h;
        error = rtFree(matrixC_pad);
        error = rtFree(workspace);
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
    __fp16 *workspace = NULL;
    if (ldc < 16)
    {
        error = rtMalloc((void **)&workspace, batch_count * N * 16 * sizeof(__fp16), RT_MEMORY_HBM);
        for (int64_t i = 0; i < batch_count; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                error = rtMemcpy(workspace + 16 * N * i + 16 * j, sizeof(__fp16) * M, matrixC + ldc * N * i + ldc * j, sizeof(__fp16) * M, RT_MEMCPY_DEVICE_TO_DEVICE);
            }
        }
        args.ldc = 16;
        args.matrixC = workspace;
        args.strideC = 16 * N;
    }
    else
    {
        args.ldc = ldc;
        args.matrixC = matrixC;
        args.strideC = strideC;
    }
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
    if (ldc < 16)
    {
        for (int64_t i = 0; i < batch_count; ++i)
        {
            for (int64_t j = 0; j < N; ++j)
            {
                error = rtMemcpy(matrixC + ldc * N * i + ldc * j, sizeof(__fp16) * ldc, workspace + 16 * N * i + 16 * j, sizeof(__fp16) * ldc, RT_MEMCPY_DEVICE_TO_DEVICE);
            }
        }
        error = rtFree(workspace);
    }
    return error;
}

rtError_t hablasHsyrk(hablasHandle_t handle,
                      hablasFillMode_t uplo,
                      hablasOperation_t transA,
                      int64_t N,
                      int64_t K,
                      __fp16 alpha,
                      __fp16 *matrixA,
                      int64_t lda,
                      __fp16 beta,
                      __fp16 *matrixC,
                      int64_t ldc)
{
    rtError_t error;
    rtStream_t stream;
    hablasGetStream(handle, &stream);
    const char *func_name = "hablas_hsyrk_kernel";
    uint64_t blockDim = CORENUM;
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
    __fp16 *workspace = NULL;
    if (ldc < 16)
    {
        error = rtMalloc((void **)&workspace, 16 * 16 * sizeof(__fp16), RT_MEMORY_HBM);
        for (int64_t i = 0; i < N; ++i)
        {
            error = rtMemcpy(workspace + 16 * i, sizeof(__fp16) * N, matrixC + ldc * i, sizeof(__fp16) * N, RT_MEMCPY_DEVICE_TO_DEVICE);
        }
        args.ldc = 16;
        args.matrixC = workspace;
    }
    else
    {
        args.ldc = ldc;
        args.matrixC = matrixC;
    }

    args.uplo = uplo;
    args.transA = transA;
    args.N = N;
    args.K = K;
    args.alpha = alpha;
    args.matrixA = matrixA;
    args.lda = lda;
    args.beta = beta;
    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);
    if (ldc < 16)
    {
        for (int64_t i = 0; i < N; ++i)
        {
            error = rtMemcpy(matrixC + ldc * i, sizeof(__fp16) * ldc, workspace + 16 * i, sizeof(__fp16) * ldc, RT_MEMCPY_DEVICE_TO_DEVICE);
        }
        error = rtFree(workspace);
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
    uint64_t blockDim = CORENUM;
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
    __fp16 *workspace = NULL;
    if (ldc < 16)
    {
        error = rtMalloc((void **)&workspace, 16 * 16 * sizeof(__fp16), RT_MEMORY_HBM);
        for (int64_t i = 0; i < N; ++i)
        {
            error = rtMemcpy(workspace + 16 * i, sizeof(__fp16) * N, matrixC + ldc * i, sizeof(__fp16) * N, RT_MEMCPY_DEVICE_TO_DEVICE);
        }
        args.ldc = 16;
        args.matrixC = workspace;
    }
    else
    {
        args.ldc = ldc;
        args.matrixC = matrixC;
    }
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

    error = rtKernelLaunch(func_name, blockDim, (void *)&args,
                           sizeof(args), NULL, stream);

    if (ldc < 16)
    {
        for (int64_t i = 0; i < N; ++i)
        {
            error = rtMemcpy(matrixC + ldc * i, sizeof(__fp16) * ldc, workspace + 16 * i, sizeof(__fp16) * ldc, RT_MEMCPY_DEVICE_TO_DEVICE);
        }
        error = rtFree(workspace);
    }
    return error;
}