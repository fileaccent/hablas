#include "hablas.h"
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

#define EXPECT_EQ(a,b) \
do {\
    assert((a == b) && "error.\n");\
} while(0)

char *ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        printf("Open file failed. path = %s", filePath.c_str());
        return nullptr;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        printf("file size is 0");
        return nullptr;
    }
    if (size > bufferSize) {
        printf("file size is large than buffer size");
        return nullptr;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    return static_cast<char *>(buffer);
}

bool compareFp16OutputData(const float *actualOutputData, const float *expectedOutputData, uint64_t len)
{
    uint64_t errorCount = 0;
    int64_t lastError = -1;
    int64_t continuous = 0;
    int64_t maxContinous = 0;
    uint64_t i = 0;
    float ratios[] = {0.001, 0.001};
    double error = 0;
    printf("cmp len:%ld\n",len);
    for (i = 0; i < len; i++) {
        float actualOutputItem = *(actualOutputData + i);
        float expectedOutputItem = *(expectedOutputData + i);
        if (i >= 0 && i < 10) {
            printf("index:%ld -> actual: %f, expected: %f\n", i, actualOutputItem, expectedOutputItem);
        }
        float tmp = abs(expectedOutputItem * ratios[0]);
        float limitError = tmp;
        if (abs((actualOutputItem - expectedOutputItem)) > limitError) {
            errorCount++;
            if (i == lastError + 1) {
                continuous++;
            } else {
                if (maxContinous < continuous) {
                    maxContinous = continuous;
                }
                continuous = 1;
            }
            lastError = i;
        }
        error = std::abs((actualOutputItem - expectedOutputItem) / expectedOutputItem);
    }
    printf("**error**:%e\n", error / len);

    if (i == len - 1) {
        if (maxContinous < continuous) {
            maxContinous = continuous;
        }
    }

    if (errorCount > len * ratios[1] || maxContinous > 16) {
        for (i = 0; i < 16; i++) {
            float actualOutputItem = *(actualOutputData + i);
            float expectedOutputItem = *(expectedOutputData + i);
            float tmp = abs(expectedOutputItem * ratios[0]);
            float limitError = tmp;
            if (abs((actualOutputItem - expectedOutputItem)) > limitError) {
                std::cout << "index:" << i << " ,cmprlst:" << abs((actualOutputItem - expectedOutputItem)) <<
                    " ,actualDataf:" << actualOutputItem << " ,expectedDataf:" <<
                    expectedOutputItem << std::endl;
            }
        }
        std::cout << "------errorCount:" << errorCount << std::endl;
        return false;
    } else {
        return true;
    }
}

int main() {

    rtError_t error = rtSetDevice(0);

    hablasHandle_t handle;
    hablasCreate(&handle);
    hablasFillMode_t mode = HABLAS_FILL_MODE_LOWER;    
    int64_t N = 128;
    int64_t lda = 128;
    int64_t incx = 2;
    int64_t incy = 3;
    int64_t sizeA = N * lda + 1;
    int64_t sizeX = N * incx;
    int64_t sizeY = N * incy;

    size_t fileSize;
	void *hA = nullptr;
	rtMallocHost(&hA, sizeA * sizeof(haComplex));
    char* fileData = ReadFile("./data/input1.bin", fileSize, hA, sizeA * sizeof(haComplex));
    if (fileData == nullptr) {
        printf("Read input1 failed");
        return 0;
    }

    void *hX = nullptr;
    rtMallocHost(&hX, sizeX * sizeof(haComplex));
    fileData = ReadFile("./data/input2.bin", fileSize, hX, sizeX * sizeof(haComplex));
    if (fileData == nullptr) {
        printf("Read input2 failed");
        return 0;
    }

    void *hY = nullptr;
    rtMallocHost(&hY, sizeY * sizeof(haComplex));
    fileData = ReadFile("./data/input3.bin", fileSize, hY, sizeY * sizeof(haComplex));
    if (fileData == nullptr) {
        printf("Read input3 failed");
        return 0;
    }

    void *hAlpha = nullptr;
    rtMallocHost(&hAlpha, sizeof(haComplex));
    fileData = ReadFile("./data/alpha.bin", fileSize, hAlpha, sizeof(haComplex));
    if (fileData == nullptr) {
        printf("Read alpha failed");
        return 0;
    }

    void *hBeta = nullptr;
    rtMallocHost(&hBeta, sizeof(haComplex));
    fileData = ReadFile("./data/beta.bin", fileSize, hBeta, sizeof(haComplex));
    if (fileData == nullptr) {
        printf("Read beta failed");
        return 0;
    }

    void *dAlpha = nullptr;
    void *dBeta = nullptr;
    void *dA = nullptr;
    void *dX = nullptr;
    void *dY = nullptr;

    error = rtMalloc((void **)&dAlpha, sizeof(haComplex), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);
    error = rtMalloc((void **)&dBeta, sizeof(haComplex), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);
    error = rtMalloc((void **)&dA, sizeA * sizeof(haComplex), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);
    error = rtMalloc((void **)&dX, sizeX * sizeof(haComplex), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);
    error = rtMalloc((void **)&dY, sizeY * sizeof(haComplex), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);

    std::cout << sizeof(haComplex) << std::endl;

    error = rtMemcpy(dAlpha,
                     sizeof(haComplex),
                     hAlpha,
                     sizeof(haComplex),
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtMemcpy(dBeta,
                     sizeof(haComplex),
                     hBeta,
                     sizeof(haComplex),
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);
    
    error = rtMemcpy(dA,
                     sizeof(haComplex)*sizeA,
                     hA,
                     sizeof(haComplex)*sizeA,
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtMemcpy(dX,
                     sizeof(haComplex)*sizeX,
                     hX,
                     sizeof(haComplex)*sizeX,
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtMemcpy(dY,
                     sizeof(haComplex)*sizeY,
                     hY,
                     sizeof(haComplex)*sizeY,
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);


    error = hablasCsymv(handle, 
                mode,
                N,
                dAlpha,
                dA,
                lda,
                dX,
                incx,
                dBeta,
                dY,
                incy);
    EXPECT_EQ(error, RT_ERROR_NONE);

    void *expect = nullptr;
    rtMallocHost(&expect, sizeY * sizeof(haComplex));
    fileData = ReadFile("./data/expect.bin", fileSize, expect, sizeY * sizeof(haComplex));
    if (fileData == nullptr) {
        printf("Read expect failed");
        return 0;
    }

    void *output = nullptr;
    rtMallocHost(&output, sizeY * sizeof(haComplex));
    error = rtMemcpy(output,
                     sizeof(haComplex)*sizeY,
                     dY,
                     sizeof(haComplex)*sizeY,
                     RT_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(error, RT_ERROR_NONE);

    bool statue = compareFp16OutputData(reinterpret_cast<const float *>(output), 
                                        reinterpret_cast<const float *>(expect),
                                        sizeY * 2);
	if(statue){
		printf("output data is same with expext!\n");
	}


    error = rtFreeHost(hAlpha);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFreeHost(hBeta);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFreeHost(hA);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFreeHost(hX);
    EXPECT_EQ(error, RT_ERROR_NONE);
	
	error = rtFreeHost(hY);
    EXPECT_EQ(error, RT_ERROR_NONE);

	
	error = rtFreeHost(expect);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFree(dAlpha);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFree(dBeta);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFree(dA);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFree(dX);
    EXPECT_EQ(error, RT_ERROR_NONE);
    
    error = rtFree(dY);
    EXPECT_EQ(error, RT_ERROR_NONE);
    return 0;
}