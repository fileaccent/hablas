#ifndef HACL_TYPE
#define HACL_TYPE

#ifdef ASCEND910A
#define CORENUM 32
#elif ASCEND910B
#define CORENUM 30
#elif ASCEND710
#define CORENUM 8
#else
#define CORENUM 32
#endif

typedef enum
{
    HABLAS_OP_N,
    HABLAS_OP_T,
    HABLAS_OP_C
} hablasOperation_t;

typedef enum
{
    HABLAS_FILL_MODE_LOWER,
    HABLAS_FILL_MODE_UPPER,
    HABLAS_FILL_MODE_FULL
} hablasFillMode_t;

typedef enum
{
    HABLAS_DIAG_NON_UNIT,
    HABLAS_DIAG_UNIT
} hablasDiagType_t;

typedef struct {
    float real;
    float imag;
} haComplex;

#endif