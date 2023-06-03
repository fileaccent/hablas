#ifndef HABDLE_H
#define HANDLE_H

#include "runtime/rt.h"

struct hablasHandle_t
{
private:
    rtStream_t stream;

public:
    friend rtError_t hablasCreate(hablasHandle_t *handle);
    friend rtError_t hablasDestroy(hablasHandle_t handle);
    friend void hablasSetStream(hablasHandle_t handle, rtStream_t streamId);
    friend void hablasGetStream(hablasHandle_t handle, rtStream_t *streamId);
};
#endif