#include "OpenCLJNI.h"
#include <vector>
#include "clApp.h"
#define printf ALOGV

jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    ALOGI("OpenCLJNI JNI_OnLoad()");
    printOpenCLInfo();
    return JNI_VERSION_1_6;
}
void JNI_OnUnload(JavaVM *vm, void *reserved) {
    ALOGI("OpenCLJNI JNI_OnUnload()");
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_vectorAdd(JNIEnv *env, jobject thiz,
        jint maxNDRange){
    ALOGI("Begin to run uBenchmark: vectorAdd");

    CCLAPP clApp(&source_vectorAdd, "vectorAdd", false);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    //Step 2: Allocate host buffers
    std::vector<float> a_host(maxNDRange, 1);
    std::vector<float> b_host(maxNDRange, 2);
    std::vector<float> c_host(maxNDRange);

    //Step 3: host >> device (Allocate device buffers and transfer data)
    /*
    cl::Buffer A_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        a_host.size() * sizeof(float), a_host.data());
    cl::Buffer B_device(clApp.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        b_host.size() * sizeof(float), b_host.data());
    cl::Buffer C_device(clApp.context, CL_MEM_READ_WRITE,
                        c_host.size() * sizeof(float));
    */
    cl_int err;
    //!here should not use CL_MEM_COPY_HOST_PTR for A and B
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, nullptr, nullptr);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, nullptr, nullptr);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), nullptr, &err);
    //err = clEnqueueWriteBuffer(clEnvironment.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, NULL, NULL);

    //Step 4: Set kernel parameters.
    //program_kernel.setArg(0, static_cast<cl_ulong>(clApp.maxNDRange));
    //program_kernel.setArg(1, A_device);
    //program_kernel.setArg(2, B_device);
    //program_kernel.setArg(3, C_device);

    err = clSetKernelArg(clApp.clKernel, 0, sizeof(int), (void*)&maxNDRange);
    err = clSetKernelArg(clApp.clKernel, 1, sizeof(cl_mem), (void*)&A_device);
    err = clSetKernelArg(clApp.clKernel, 2, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel, 3, sizeof(cl_mem), (void*)&C_device);

    //Step 5: Launch kernel on the compute device.
    //clApp.queue.enqueueNDRangeKernel(program_kernel, cl::NullRange, clApp.maxNDRange, cl::NullRange);
    //clApp.queue.finish();//block host until device finishes

    size_t global_size[1] = {(size_t)maxNDRange};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 1
            , nullptr
            , global_size
            , nullptr
            , 0
            , nullptr
            , nullptr);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    //clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, nullptr, nullptr);

    int maxPrintSize = 16;
    int n = maxNDRange > maxPrintSize ? maxPrintSize : maxNDRange;
    printf("OpenCL calculation done. Total %d elements. Print the first %d numbers. \n", maxNDRange, n);
    std::string strA, strB, strC;
    for(int i = 0; i < n; i++) {
        strA += (std::to_string(a_host[i]) + ", ");
        strB += (std::to_string(b_host[i]) + ", ");
        strC += (std::to_string(c_host[i]) + ", ");
    }
    printf("Input A is %s", strA.c_str());
    printf("Input B is %s", strB.c_str());
    printf("Output C is %s", strC.c_str());

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(C_device);

    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_matrixMul1(JNIEnv *env, jobject thiz,
        jint DIM, jint TILESIZE){
    ALOGI("Begin to run uBenchmark: matrixMul1");

    CCLAPP clApp(&source_matrixMul, "matrixMul1", false);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    //Step 2: Allocate host buffers
    size_t matrixDimM = DIM;
    size_t matrixDimK = DIM;
    size_t matrixDimN = DIM;
    std::vector<float> a_host(matrixDimM*matrixDimK, 1);
    std::vector<float> b_host(matrixDimK*matrixDimN, 2);
    std::vector<float> c_host(matrixDimM*matrixDimN);

    for (int i=0; i<matrixDimM*matrixDimK; i++)
        a_host[i] = (float)rand() / (float)RAND_MAX;
    for (int i=0; i<matrixDimK*matrixDimN; i++)
        b_host[i] = (float)rand() / (float)RAND_MAX;

    //Step 3: host >> device (Allocate device buffers and transfer data)
    cl_int err;
    //!here should not use CL_MEM_COPY_HOST_PTR for A and B
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, nullptr, nullptr);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, nullptr, nullptr);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), nullptr, &err);

    //Step 4: Set kernel parameters.
    err = clSetKernelArg(clApp.clKernel, 0, sizeof(int), (void*)&matrixDimM);
    err = clSetKernelArg(clApp.clKernel, 1, sizeof(int), (void*)&matrixDimN);
    err = clSetKernelArg(clApp.clKernel, 2, sizeof(int), (void*)&matrixDimK);
    err = clSetKernelArg(clApp.clKernel, 3, sizeof(cl_mem), (void*)&A_device);
    err = clSetKernelArg(clApp.clKernel, 4, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel, 5, sizeof(cl_mem), (void*)&C_device);

    //Step 5: Launch kernel on the compute device.
    size_t local[2] = {(size_t)TILESIZE, (size_t)TILESIZE};
    size_t global[2] = {matrixDimM, matrixDimN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 2
            , nullptr
            , global
            , local
            , 0
            , nullptr
            , nullptr);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, nullptr, nullptr);

    int totalSize = DIM * DIM;
    int maxPrintSize = 16;
    int n = totalSize > maxPrintSize ? maxPrintSize : totalSize;
    printf("OpenCL calculation done. Total %d elements. Print the first %d numbers. \n", totalSize, n);
    std::string strA, strB, strC;
    for(int i = 0; i < n; i++) {
        strA += (std::to_string(a_host[i]) + ", ");
        strB += (std::to_string(b_host[i]) + ", ");
        strC += (std::to_string(c_host[i]) + ", ");
    }
    printf("Input A is %s", strA.c_str());
    printf("Input B is %s", strB.c_str());
    printf("Output C is %s", strC.c_str());

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(C_device);

    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_matrixMul5(JNIEnv *env, jobject thiz,
        jint DIM, jint TILESIZE, jint WPT, jint TRANSPOSEX, jint TRANSPOSEY){
    ALOGI("Begin to run uBenchmark: matrixMul5");

    CCLAPP clApp(&source_matrixMul, "matrixMul5", true);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    //Step 2: Allocate host buffers
    size_t matrixDimM = DIM;
    size_t matrixDimK = DIM;
    size_t matrixDimN = DIM;
    std::vector<float> a_host(matrixDimM*matrixDimK, 1);
    std::vector<float> b_host(matrixDimK*matrixDimN, 2);
    std::vector<float> c_host(matrixDimM*matrixDimN);

    for (int i=0; i<matrixDimM*matrixDimK; i++)
        a_host[i] = (float)rand() / (float)RAND_MAX;
    for (int i=0; i<matrixDimK*matrixDimN; i++)
        b_host[i] = (float)rand() / (float)RAND_MAX;

    //Step 3: host >> device (Allocate device buffers and transfer data)
    cl_int err;
    //!here should not use CL_MEM_COPY_HOST_PTR for A and B
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, nullptr, nullptr);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, nullptr, nullptr);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), nullptr, &err);

    //if Transpose == true
    cl_mem B_TR_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), nullptr, &err);


    //Step 4: Set kernel parameters.
    //if Transpose == true
    err = clSetKernelArg(clApp.clKernel_transpose, 0, sizeof(int), (void*)&matrixDimK);
    err = clSetKernelArg(clApp.clKernel_transpose, 1, sizeof(int), (void*)&matrixDimN);
    err = clSetKernelArg(clApp.clKernel_transpose, 2, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel_transpose, 3, sizeof(cl_mem), (void*)&B_TR_device);

    err = clSetKernelArg(clApp.clKernel, 0, sizeof(int), (void*)&matrixDimM);
    err = clSetKernelArg(clApp.clKernel, 1, sizeof(int), (void*)&matrixDimN);
    err = clSetKernelArg(clApp.clKernel, 2, sizeof(int), (void*)&matrixDimK);
    err = clSetKernelArg(clApp.clKernel, 3, sizeof(cl_mem), (void*)&A_device);
    //if Transpose == true
    //err = clSetKernelArg(clEnvironment.clKernel, 4, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel, 4, sizeof(cl_mem), (void*)&B_TR_device);
    err = clSetKernelArg(clApp.clKernel, 5, sizeof(cl_mem), (void*)&C_device);

    //Step 5: Launch kernel on the compute device.
    //if Transpose == true
    size_t local_TR[2] = {(size_t)TRANSPOSEX, (size_t)TRANSPOSEY};
    size_t global_TR[2] = {matrixDimK, matrixDimN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel_transpose
            , 2
            , nullptr
            , global_TR
            , local_TR
            , 0
            , nullptr
            , nullptr);
    size_t local[2] = {(size_t)TILESIZE, (size_t)(TILESIZE/WPT)};
    size_t global[2] = {matrixDimM, matrixDimN/WPT};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 2
            , nullptr
            , global
            , local
            , 0
            , nullptr
            , nullptr);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, nullptr, nullptr);

    int totalSize = DIM * DIM;
    int maxPrintSize = 16;
    int n = totalSize > maxPrintSize ? maxPrintSize : totalSize;
    printf("OpenCL calculation done. Total %d elements. Print the first %d numbers. \n", totalSize, n);
    std::string strA, strB, strC;
    for(int i = 0; i < n; i++) {
        strA += (std::to_string(a_host[i]) + ", ");
        strB += (std::to_string(b_host[i]) + ", ");
        strC += (std::to_string(c_host[i]) + ", ");
    }
    printf("Input A is %s", strA.c_str());
    printf("Input B is %s", strB.c_str());
    printf("Output C is %s", strC.c_str());

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(B_TR_device);
    clReleaseMemObject(C_device);

    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_matrixMul6(JNIEnv *env, jobject thiz,
           jint DIM, jint TSM, jint TSN, jint WPTM, jint WPTN, jint TRANSPOSEX, jint TRANSPOSEY){
    ALOGI("Begin to run uBenchmark: matrixMul6");

    CCLAPP clApp(&source_matrixMul, "matrixMul6", true);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    //Step 2: Allocate host buffers
    size_t matrixDimM = DIM;
    size_t matrixDimK = DIM;
    size_t matrixDimN = DIM;
    std::vector<float> a_host(matrixDimM*matrixDimK, 1);
    std::vector<float> b_host(matrixDimK*matrixDimN, 2);
    std::vector<float> c_host(matrixDimM*matrixDimN);

    for (int i=0; i<matrixDimM*matrixDimK; i++)
        a_host[i] = (float)rand() / (float)RAND_MAX;
    for (int i=0; i<matrixDimK*matrixDimN; i++)
        b_host[i] = (float)rand() / (float)RAND_MAX;

    //Step 3: host >> device (Allocate device buffers and transfer data)
    cl_int err;
    //!here should not use CL_MEM_COPY_HOST_PTR for A and B
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, nullptr, nullptr);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), nullptr, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, nullptr, nullptr);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), nullptr, &err);

    //if Transpose == true
    cl_mem B_TR_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), nullptr, &err);

    //Step 4: Set kernel parameters.
    //if Transpose == true
    err = clSetKernelArg(clApp.clKernel_transpose, 0, sizeof(int), (void*)&matrixDimK);
    err = clSetKernelArg(clApp.clKernel_transpose, 1, sizeof(int), (void*)&matrixDimN);
    err = clSetKernelArg(clApp.clKernel_transpose, 2, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel_transpose, 3, sizeof(cl_mem), (void*)&B_TR_device);

    err = clSetKernelArg(clApp.clKernel, 0, sizeof(int), (void*)&matrixDimM);
    err = clSetKernelArg(clApp.clKernel, 1, sizeof(int), (void*)&matrixDimN);
    err = clSetKernelArg(clApp.clKernel, 2, sizeof(int), (void*)&matrixDimK);
    err = clSetKernelArg(clApp.clKernel, 3, sizeof(cl_mem), (void*)&A_device);
    //if Transpose == true
    //err = clSetKernelArg(clEnvironment.clKernel, 4, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel, 4, sizeof(cl_mem), (void*)&B_TR_device);
    err = clSetKernelArg(clApp.clKernel, 5, sizeof(cl_mem), (void*)&C_device);

    //Step 5: Launch kernel on the compute device.
    //if Transpose == true
    size_t local_TR[2] = {(size_t)TRANSPOSEX, (size_t)TRANSPOSEY};
    size_t global_TR[2] = {matrixDimK, matrixDimN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel_transpose
            , 2
            , nullptr
            , global_TR
            , local_TR
            , 0
            , nullptr
            , nullptr);

    //if kernel 6
    size_t local[2] = {(size_t)(TSM/WPTM), (size_t)(TSN/WPTN)};
    size_t global[2] = {matrixDimM/WPTM, matrixDimN/WPTN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 2
            , nullptr
            , global
            , local
            , 0
            , nullptr
            , nullptr);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, nullptr, nullptr);

    int totalSize = DIM * DIM;
    int maxPrintSize = 16;
    int n = totalSize > maxPrintSize ? maxPrintSize : totalSize;
            printf("OpenCL calculation done. Total %d elements. Print the first %d numbers. \n", totalSize, n);
    std::string strA, strB, strC;
    for(int i = 0; i < n; i++) {
        strA += (std::to_string(a_host[i]) + ", ");
        strB += (std::to_string(b_host[i]) + ", ");
        strC += (std::to_string(c_host[i]) + ", ");
    }
    printf("Input A is %s", strA.c_str());
    printf("Input B is %s", strB.c_str());
    printf("Output C is %s", strC.c_str());

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(B_TR_device);
    clReleaseMemObject(C_device);

    return true;
}

