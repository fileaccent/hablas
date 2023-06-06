#ifndef HABLAS_H
#define HABLAS_H

#include "runtime/rt.h"
#include "hacl_type.h"
#include "handle.h"

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
                      int64_t ldc);

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
                             int64_t batch_count);

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
                                    int64_t batch_count);

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
                      int64_t ldc);

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
                       int64_t ldc);

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
                      int64_t incy);

rtError_t hablasSgemv(
                   hablasHandle_t handle,
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
                   int64_t incy);

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
                      int64_t incy);

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
                      int64_t incy);
#endif