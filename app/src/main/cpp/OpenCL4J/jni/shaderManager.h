//
// Created by Xiaojun on 9/12/2023.
//

#ifndef ANDROIDOPENCL_SHADERMANAGER_H
#define ANDROIDOPENCL_SHADERMANAGER_H

static const char * source_matrixMul =
        "kernel void matrixMul1(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){\n"
        "   const int globalRow = get_global_id(0);\n"
        "   const int globalCol = get_global_id(1);\n"
        "   float acc = 0.0f;\n"
        "   for (int k=0; k<K; k++) {\n"
        "      acc += A[k*M + globalRow] * B[globalCol*K + k];\n"
        "   }\n"
        "   C[globalCol*M + globalRow] = acc;\n"
        "}";

static const char * source =
        "__kernel void rgba_to_gray(__read_only image2d_t input, __write_only image2d_t output)\n"
        "{\n"
        "    int2 coord = (int2)(get_global_id(0), get_global_id(1));\n"
        "    float4 pixel = read_imagef(input, CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP|CLK_FILTER_NEAREST, coord);\n"
        "    float gray = dot(pixel.xyz, (float3)(0.299, 0.587, 0.114));\n"
        "    write_imagef(output, coord, (float4)(gray, gray, gray, pixel.w));\n"
        "}";

static const char * source_vectorAdd =
        "__kernel void vectorAdd(\n"
        "        const int n,\n"
        "        const __global float *a, \n"
        "        const __global float *b, \n"
        "        __global float *c \n"
        "        )\n"
        "{\n"
        "   const int i = get_global_id(0);\n"
        "   if (i < n) {\n"
        "       c[i] = a[i] + b[i];\n"
        "   }\n"
        "}";






#endif //ANDROIDOPENCL_SHADERMANAGER_H
