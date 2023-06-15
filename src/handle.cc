#include "handle.h"

rtError_t hablasCreate(hablasHandle_t *handle)
{
    rtError_t error = rtStreamCreate(&(handle->stream), 0);
    return error;
}

rtError_t hablasDestroy(hablasHandle_t handle)
{
    rtError_t error = rtStreamDestroy(handle.stream);
    return error;
}

void hablasSetStream(hablasHandle_t handle, rtStream_t streamId)
{
    handle.stream = streamId;
}

void hablasGetStream(hablasHandle_t handle, rtStream_t *streamId)
{
    *streamId = handle.stream;
}