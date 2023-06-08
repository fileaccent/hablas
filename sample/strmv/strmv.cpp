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
    hablasOperation_t transa = HABLAS_OP_N;    
    hablasDiagType_t diag = HABLAS_DIAG_NON_UNIT;
    int64_t N = 129;
    int64_t lda = 130;
    int64_t incx = 2;
    int64_t sizeA = N * lda;
    int64_t sizeX = N * incx;

    size_t fileSize;

	void *hA = nullptr;
	rtMallocHost(&hA, sizeA * sizeof(float));
    char* fileData = ReadFile("./data/input1.bin", fileSize, hA, sizeA * sizeof(float));
    if (fileData == nullptr) {
        printf("Read input1 failed");
        return 0;
    }

    void *hX = nullptr;
    rtMallocHost(&hX, sizeX * sizeof(float));
    fileData = ReadFile("./data/input2.bin", fileSize, hX, sizeX * sizeof(float));
    if (fileData == nullptr) {
        printf("Read input2 failed");
        return 0;
    }

    void *dA = nullptr;
    void *dX = nullptr;

    error = rtMalloc((void **)&dA, sizeA * sizeof(float), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);
    error = rtMalloc((void **)&dX, sizeX * sizeof(float), RT_MEMORY_HBM);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtMemcpy(dA,
                     sizeof(float)*sizeA,
                     hA,
                     sizeof(float)*sizeA,
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtMemcpy(dX,
                     sizeof(float)*sizeX,
                     hX,
                     sizeof(float)*sizeX,
                     RT_MEMCPY_HOST_TO_DEVICE);
    EXPECT_EQ(error, RT_ERROR_NONE);

    hablasStrmv(handle, 
                mode,
                transa,
                diag,
                N,
                dA,
                lda,
                dX,
                incx);
    
    void *expect = nullptr;
    rtMallocHost(&expect, sizeX * sizeof(float));
    fileData = ReadFile("./data/expect.bin", fileSize, expect, sizeX * sizeof(float));
    if (fileData == nullptr) {
        printf("Read expect failed");
        return 0;
    }

    void *output = nullptr;
    rtMallocHost(&output, sizeX * sizeof(float));
    error = rtMemcpy(output,
                     sizeof(float)*sizeX,
                     dX,
                     sizeof(float)*sizeX,
                     RT_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(error, RT_ERROR_NONE);

    bool statue = compareFp16OutputData(reinterpret_cast<const float *>(output), 
                                        reinterpret_cast<const float *>(expect),
                                        sizeX);
	if(statue){
		printf("output data is same with expext!\n");
	}

    error = rtFreeHost(hA);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFreeHost(hX);
    EXPECT_EQ(error, RT_ERROR_NONE);
	
    error = rtFree(dA);
    EXPECT_EQ(error, RT_ERROR_NONE);

    error = rtFree(dX);
    EXPECT_EQ(error, RT_ERROR_NONE);
    
    return 0;
}