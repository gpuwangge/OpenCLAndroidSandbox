//
// Created by Xiaojun on 9/12/2023.
//

#ifndef ANDROIDOPENCL_CLAPP_H
#define ANDROIDOPENCL_CLAPP_H

//#include "CL/cl_gl.h"
//#include "CL/cl_egl.h"
#include "CL/cl.h"
#include "AndroidLog.hpp"
#include "string"
#define printf ALOGV

class CCLAPP {
public:
    CCLAPP(const char** source, const char* name);

    ~CCLAPP();

    bool ready = false;
    cl_context clContext = nullptr;
    cl_command_queue clCommandQueue = nullptr;
    cl_program clProgram = nullptr;
    cl_kernel clKernel = nullptr;
};

void printOpenCLInfo();

#endif //ANDROIDOPENCL_CLAPP_H
