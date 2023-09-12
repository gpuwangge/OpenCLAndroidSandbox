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
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_vectorAdd(JNIEnv *env, jobject thiz){
    ALOGI("Begin to run uBenchmark: vectorAdd");

    CCLAPP clApp(&source_vectorAdd, "vectorAdd", false);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    //Step 2: Allocate host buffers
    size_t maxNDRange = 1 << 5;
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
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, NULL, NULL);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, NULL, NULL);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), NULL, &err);
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

    size_t global_size[1] = {maxNDRange};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 1
            , NULL
            , global_size
            , NULL
            , 0
            , NULL
            , NULL);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    //clApp.queue.enqueueReadBuffer(C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data());
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, NULL, NULL);
    printf("OpenCL calculation done. Result is %f,%f,%f,%f,%f \n", c_host[0],c_host[1],c_host[2],c_host[3],c_host[4]);

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(C_device);

    return false;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_matrixMul1(JNIEnv *env, jobject thiz){
    ALOGI("Begin to run uBenchmark: matrixMul1");

    CCLAPP clApp(&source_matrixMul, "matrixMul1", false);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    const int DIM = 3;
    const int TILESIZE = 1;

    //Step 2: Allocate host buffers
    const int matrixDimM = DIM;
    const int matrixDimK = DIM;
    const int matrixDimN = DIM;
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
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, NULL, NULL);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, NULL, NULL);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), NULL, &err);

    //Step 4: Set kernel parameters.
    err = clSetKernelArg(clApp.clKernel, 0, sizeof(int), (void*)&matrixDimM);
    err = clSetKernelArg(clApp.clKernel, 1, sizeof(int), (void*)&matrixDimN);
    err = clSetKernelArg(clApp.clKernel, 2, sizeof(int), (void*)&matrixDimK);
    err = clSetKernelArg(clApp.clKernel, 3, sizeof(cl_mem), (void*)&A_device);
    err = clSetKernelArg(clApp.clKernel, 4, sizeof(cl_mem), (void*)&B_device);
    err = clSetKernelArg(clApp.clKernel, 5, sizeof(cl_mem), (void*)&C_device);

    //Step 5: Launch kernel on the compute device.
    size_t local[2] = {TILESIZE, TILESIZE};
    size_t global[2] = {matrixDimM, matrixDimN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 2
            , NULL
            , global
            , local
            , 0
            , NULL
            , NULL);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, NULL, NULL);

    printf("OpenCL calculation done. Print the first 9 numbers. \n");
    printf("Input A is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", a_host[0],a_host[1],a_host[2],a_host[3],a_host[4],a_host[5],a_host[6],a_host[7],a_host[8]);
    printf("Input B is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", b_host[0],b_host[1],b_host[2],b_host[3],b_host[4],b_host[5],b_host[6],b_host[7],b_host[8]);
    printf("Output C is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", c_host[0],c_host[1],c_host[2],c_host[3],c_host[4],c_host[5],c_host[6],c_host[7],c_host[8]);

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(C_device);

    return false;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_matrixMul5(JNIEnv *env, jobject thiz){
    ALOGI("Begin to run uBenchmark: matrixMul5");

    CCLAPP clApp(&source_matrixMul, "matrixMul5", true);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    const int DIM = 3;
    const int TILESIZE = 1;

    //Step 2: Allocate host buffers
    const int matrixDimM = DIM;
    const int matrixDimK = DIM;
    const int matrixDimN = DIM;
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
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, NULL, NULL);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, NULL, NULL);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), NULL, &err);

    //if Transpose == true
    cl_mem B_TR_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), NULL, &err);


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
    const int TRANSPOSEX = 1;
    const int TRANSPOSEY = 1;
    size_t local_TR[2] = {TRANSPOSEX, TRANSPOSEY};
    size_t global_TR[2] = {matrixDimK, matrixDimN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel_transpose
            , 2
            , NULL
            , global_TR
            , local_TR
            , 0
            , NULL
            , NULL);
    const int WPT = 1;
    size_t local[2] = {TILESIZE, TILESIZE/WPT};
    size_t global[2] = {matrixDimM, matrixDimN/WPT};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 2
            , NULL
            , global
            , local
            , 0
            , NULL
            , NULL);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, NULL, NULL);

    printf("OpenCL calculation done. Print the first 9 numbers. \n");
    printf("Input A is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", a_host[0],a_host[1],a_host[2],a_host[3],a_host[4],a_host[5],a_host[6],a_host[7],a_host[8]);
    printf("Input B is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", b_host[0],b_host[1],b_host[2],b_host[3],b_host[4],b_host[5],b_host[6],b_host[7],b_host[8]);
    printf("Output C is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", c_host[0],c_host[1],c_host[2],c_host[3],c_host[4],c_host[5],c_host[6],c_host[7],c_host[8]);

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(C_device);

    return false;
}

extern "C"
JNIEXPORT jboolean JNICALL Java_com_wangge_opencl_jni_uBenchmarkManager_matrixMul6(JNIEnv *env, jobject thiz){
    ALOGI("Begin to run uBenchmark: matrixMul6");

    CCLAPP clApp(&source_matrixMul, "matrixMul6", true);
    if (!clApp.ready) {
        printf("clApp is NOT ready\n");
        return false;
    }

    const int DIM = 3;
    const int TILESIZE = 1;

    //Step 2: Allocate host buffers
    const int matrixDimM = DIM;
    const int matrixDimK = DIM;
    const int matrixDimN = DIM;
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
    cl_mem A_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  a_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, A_device, CL_TRUE, 0, a_host.size() * sizeof(float), a_host.data(), 0, NULL, NULL);
    cl_mem B_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), NULL, &err);
    err = clEnqueueWriteBuffer(clApp.clCommandQueue, B_device, CL_TRUE, 0, b_host.size() * sizeof(float), b_host.data(), 0, NULL, NULL);
    cl_mem C_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_WRITE,  c_host.size() * sizeof(float), NULL, &err);

    //if Transpose == true
    cl_mem B_TR_device    = clCreateBuffer(clApp.clContext, CL_MEM_READ_ONLY,  b_host.size() * sizeof(float), NULL, &err);


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
    const int TRANSPOSEX = 1;
    const int TRANSPOSEY = 1;
    size_t local_TR[2] = {TRANSPOSEX, TRANSPOSEY};
    size_t global_TR[2] = {matrixDimK, matrixDimN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel_transpose
            , 2
            , NULL
            , global_TR
            , local_TR
            , 0
            , NULL
            , NULL);

    //if kernel 6
    const int TSM = 1;
    const int TSN = 1;
    const int WPTM = 1;
    const int WPTN = 1;
    size_t local[2] = {TSM/WPTM, TSN/WPTN};
    size_t global[2] = {matrixDimM/WPTM, matrixDimN/WPTN};
    clEnqueueNDRangeKernel(clApp.clCommandQueue
            , clApp.clKernel
            , 2
            , NULL
            , global
            , local
            , 0
            , NULL
            , NULL);
    clFinish(clApp.clCommandQueue);

    //Step 6: device >> host
    err = clEnqueueReadBuffer(clApp.clCommandQueue, C_device, CL_TRUE, 0, c_host.size() * sizeof(float), c_host.data(), 0, NULL, NULL);

    printf("OpenCL calculation done. Print the first 9 numbers. \n");
    printf("Input A is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", a_host[0],a_host[1],a_host[2],a_host[3],a_host[4],a_host[5],a_host[6],a_host[7],a_host[8]);
    printf("Input B is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", b_host[0],b_host[1],b_host[2],b_host[3],b_host[4],b_host[5],b_host[6],b_host[7],b_host[8]);
    printf("Output C is %f,%f,%f,%f,%f,%f,%f,%f,%f \n", c_host[0],c_host[1],c_host[2],c_host[3],c_host[4],c_host[5],c_host[6],c_host[7],c_host[8]);

    clReleaseMemObject(A_device);
    clReleaseMemObject(B_device);
    clReleaseMemObject(C_device);

    return false;
}

