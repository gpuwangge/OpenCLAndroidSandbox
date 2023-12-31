//
// Created by Xiaojun on 9/12/2023.
//
#include "clApp.h"

CCLAPP::CCLAPP(const char** source, const char* name, bool bTranspose){
    //printf("CClApp Constructor\n");
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_platform_id platformId = nullptr;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_device_id deviceId = nullptr;
    clGetPlatformIDs(0, nullptr, &platformCount);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)* platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
    for (cl_uint i = 0; i < platformCount; i++) {
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
        value = (char*)malloc(valueSize);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, valueSize, value, NULL);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        printf("%d. OpenCL Platform [%d]: CL_PLATFORM_NAME: %s; Total Device Count: %d\n", i + 1, i + 1, value, deviceCount);
        free(value);

        // get all GPU devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id)* deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (cl_uint j = 0; j < deviceCount; j++) {
            // print device entexsion
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, valueSize, value, NULL);
            printf("CL_DEVICE_EXTENSIONS: %s\n", value);
            if (strpbrk(value, "cl_khr_gl_sharing") != nullptr
                && strpbrk(value, "cl_khr_egl_event") != nullptr) {
                platformId = platforms[i];
                deviceId = devices[j];
                break;
            }
            free(value);
        }

    }
    if (deviceId == nullptr) {
        return;
    }

    //EGLContext eglContext = eglGetCurrentContext();
    //EGLDisplay eglDisplay = eglGetCurrentDisplay();
    //printf("    Creating Context... \n");
    cl_context_properties props[] = {//CL_GL_CONTEXT_KHR,
            //(cl_context_properties) eglContext,
            //CL_EGL_DISPLAY_KHR,
            //(cl_context_properties) eglDisplay,
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) platformId,
            CL_NONE};
    // https://man.opencl.org/clCreateContext.html
    int error_code;
    clContext = clCreateContext(props
            , 1
            , &deviceId
            , nullptr
            , nullptr
            , &error_code);
    if (!clContext) {
        ALOGE("Create cl_context error: %d", error_code);
        return;
    }
    //printf("    Creating CommandQueue... \n");
    clCommandQueue = clCreateCommandQueue(clContext
            , deviceId
            , CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
            , &error_code);
    if (!clCommandQueue) {
        ALOGE("Create cl_command_queue error: %d", error_code);
        return;
    }
    //printf("    Creating Program With Source... \n");
    clProgram = clCreateProgramWithSource(clContext
            , 1
            , source
            , nullptr
            , &error_code);
    if (!clProgram) {
        ALOGE("Create cl_program error: %d", error_code);
        return;
    }
    //printf("    Building Program... \n");
    cl_program_info clBuildProgramResult = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
    if (clBuildProgramResult != CL_SUCCESS) {
        clGetProgramBuildInfo(clProgram
                , deviceId
                , CL_PROGRAM_BUILD_LOG
                , 0
                , nullptr
                , &valueSize);
        value = (char*) malloc(valueSize);
        clGetProgramBuildInfo(clProgram
                , deviceId
                , CL_PROGRAM_BUILD_LOG
                , valueSize
                , value
                , nullptr);
        ALOGI("Build cl_program info: %s", value);
        free(value);
        return;
    }

    //printf("    Creating kernel... \n");
    clKernel = clCreateKernel(clProgram, name, &error_code);
    if (!clKernel) {
        ALOGE("Create cl_kernel error: %d", error_code);
        return;
    }
    if(bTranspose){
        //printf("    Creating transpose kernel... \n");
        clKernel_transpose = clCreateKernel(clProgram, "transpose", &error_code);
        if (!clKernel_transpose) {
            ALOGE("Create cl_kernel_transpose error: %d", error_code);
            return;
        }
    }

    printf("OpenCL is Ready!\n");
    ready = true;
}

CCLAPP::~CCLAPP(){
    //printf("CClApp ~Constructor\n");
    if (clProgram != nullptr) {
        clReleaseProgram(clProgram);
    }
    if (clCommandQueue != nullptr) {
        clReleaseCommandQueue(clCommandQueue);
    }
    if (clKernel != nullptr) {
        clReleaseKernel(clKernel);
    }
    if (clKernel_transpose != nullptr) {
        clReleaseKernel(clKernel_transpose);
    }
    if (clContext != nullptr) {
        clReleaseContext(clContext);
    }
}



void printOpenCLInfo() {
    int i, j;
    char* value;
    size_t valueSize;
    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    cl_uint maxComputeUnits;
    cl_uint maxClockFreq;
    unsigned int maxWorkItemDimension;
    size_t maxWorkItemSize[3];
    size_t maxWorkGroupSize;
    cl_uint devAddressBit;
    cl_uint memBaseAddressAlign;
    cl_ulong maxMemAllocSize;
    cl_ulong gloMemSize;
    cl_ulong maxConstBuffSize;
    cl_ulong gloMemCacheSize;
    cl_uint gloMemCacheLineSize;
    cl_ulong localMemSize;
    size_t profTimerRes;
    cl_bool imageSupport;
    cl_bool errorCorrSupport;
    cl_bool hostUnifiedMem;
    cl_uint preVecWidthInt;
    cl_uint preVecWidthLong;
    cl_uint preVecWidthFloat;
    cl_uint preVecWidthDouble;
    cl_uint nativeVecWidthInt;
    cl_uint nativeVecWidthLong;
    cl_uint nativeVecWidthFloat;
    cl_uint nativeVecWidthDouble;

    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)* platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);

            printf("There is a total of %d OpenCL platforms.\n", platformCount);

    for (i = 0; i < platformCount; i++) {

                printf("%d. OpenCL Platform [%d]:\n", i + 1, i + 1);

        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
        value = (char*)malloc(valueSize);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, valueSize, value, NULL);
                printf("   CL_PLATFORM_NAME: %s\n", value);
        free(value);

        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
                printf("   Total Device Count: %d\n", deviceCount);

        // get all GPU devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        devices = (cl_device_id*)malloc(sizeof(cl_device_id)* deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, devices, NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++) {

                    printf("   GPU Device [%d]: \n", j+1);

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
                    printf("    CL_DEVICE_NAME: %s\n", value);
            free(value);

            // print device vendor
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, valueSize, value, NULL);
                    printf("    CL_DEVICE_VENDOR: %s\n", value);
            free(value);

            // print device profile
            clGetDeviceInfo(devices[j], CL_DEVICE_PROFILE, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_PROFILE, valueSize, value, NULL);
                    printf("    CL_DEVICE_PROFILE: %s\n", value);
            free(value);

            // print device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
                    printf("    CL_DEVICE_VERSION: %s\n", value);
            free(value);

            // print driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
                    printf("    CL_DRIVER_VERSION: %s\n", value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
                    printf("    CL_DEVICE_OPENCL_C_VERSION: %s\n", value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
                    printf("    CL_DEVICE_MAX_COMPUTE_UNITS: %d\n", maxComputeUnits);

            // print maximum clock frequency
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &maxClockFreq, NULL);
                    printf("    CL_DEVICE_MAX_CLOCK_FREQUENCY: %d MHz\n", maxClockFreq);

            // print maximum work item dimension
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(unsigned int), &maxWorkItemDimension, NULL);
                    printf("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %d\n", maxWorkItemDimension);

            // print maximum work item size
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*maxWorkItemDimension, &maxWorkItemSize, NULL);
                    printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES: (%d, %d, %d)\n", maxWorkItemSize[0], maxWorkItemSize[1], maxWorkItemSize[2]);

            // print maximum work group size
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
                    printf("    CL_DEVICE_MAX_WORK_GROUP_SIZES: %d\n", maxWorkGroupSize);

            // print device address bits
            clGetDeviceInfo(devices[j], CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &devAddressBit, NULL);
                    printf("    CL_DEVICE_ADDRESS_BITS: %d\n", devAddressBit);

            // print device memory base address align
            clGetDeviceInfo(devices[j], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &memBaseAddressAlign, NULL);
                    printf("    CL_DEVICE_MEM_BASE_ADDR_ALIGN: %d\n", memBaseAddressAlign);

            // print device max memory allocation size
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxMemAllocSize, NULL);
                    printf("    CL_DEVICE_MAX_MEM_ALLOC_SIZE: %ld\n", maxMemAllocSize);

            // print device global memry size
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &gloMemSize, NULL);
                    printf("    CL_DEVICE_GLOBAL_MEM_SIZE: %ld\n", gloMemSize);

            // print device maximum constant buffer size
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), &maxConstBuffSize, NULL);
                    printf("    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: %ld\n", maxConstBuffSize);

            // print device global memory cache size
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &gloMemCacheSize, NULL);
                    printf("    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: %ld\n", gloMemCacheSize);

            // print device global memory cacheline size
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), &gloMemCacheLineSize, NULL);
                    printf("    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: %d\n", gloMemCacheLineSize);

            // print device local memory size
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
                    printf("    CL_DEVICE_LOCAL_MEM_SIZE: %ld\n", localMemSize);

            // print device profiling timer resoultion
            clGetDeviceInfo(devices[j], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), &profTimerRes, NULL);
                    printf("    CL_DEVICE_PROFILING_TIMER_RESOLUTION: %d\n", profTimerRes);

            // print device image support
            clGetDeviceInfo(devices[j], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, NULL);
                    printf("    CL_DEVICE_IMAGE_SUPPORT: %d\n", imageSupport);

            // print device error correction support
            clGetDeviceInfo(devices[j], CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(cl_bool), &errorCorrSupport, NULL);
                    printf("    CL_DEVICE_ERROR_CORRECTION_SUPPORT: %d\n", errorCorrSupport);

            // print device error correction support
            clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &hostUnifiedMem, NULL);
                    printf("    CL_DEVICE_HOST_UNIFIED_MEMORY: %d\n", hostUnifiedMem);

            // print device entexsion
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, 0, NULL, &valueSize);
            value = (char*)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_EXTENSIONS, valueSize, value, NULL);
                    printf("    CL_DEVICE_EXTENSIONS: %s\n", value);
            free(value);

            // print device preferred vector width int
            clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &preVecWidthInt, NULL);
                    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: %d\n", preVecWidthInt);

            // print device preferred vector width long
            clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &preVecWidthLong, NULL);
                    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: %d\n", preVecWidthLong);

            // print device preferred vector width float
            clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &preVecWidthFloat, NULL);
                    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: %d\n", preVecWidthFloat);

            // print device preferred vector width double
            clGetDeviceInfo(devices[j], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &preVecWidthDouble, NULL);
                    printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: %d\n", preVecWidthDouble);

            // print device native vector width int
            clGetDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(cl_uint), &nativeVecWidthInt, NULL);
                    printf("    CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: %d\n", nativeVecWidthInt);

            // print device native vector width long
            clGetDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, sizeof(cl_uint), &nativeVecWidthLong, NULL);
                    printf("    CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: %d\n", nativeVecWidthLong);

            // print device native vector width float
            clGetDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &nativeVecWidthFloat, NULL);
                    printf("    CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: %d\n", nativeVecWidthFloat);

            // print device native vector width double
            clGetDeviceInfo(devices[j], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &nativeVecWidthDouble, NULL);
                    printf("    CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: %d\n", nativeVecWidthDouble);

                    printf("\n");
                    printf("------------------------------------------------------------------------------\n\n");
        }

        free(devices);

    }
    free(platforms);
}

