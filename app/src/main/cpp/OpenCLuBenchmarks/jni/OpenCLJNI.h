#ifndef ANDROIDOPENCL_OPENCLJNI_H
#define ANDROIDOPENCL_OPENCLJNI_H

#include "jni.h"
#include "CL/cl.h"
#include "AndroidLog.hpp"
#include "shaderManager.h"

#ifdef __cplusplus
extern "C"
{
#endif
    extern JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved);
    extern JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved);
#ifdef __cplusplus
}
#endif

#endif //ANDROIDOPENCL_OPENCLJNI_H
