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
        "}\n"

        "#define TRANSPOSEX 1\n"
        "#define TRANSPOSEY 1\n"
        "kernel void transpose(const int P, const int Q, global const float* input,  global float* output) {\n"
        "       const int tx = get_local_id(0);\n"
        "       const int ty = get_local_id(1);\n"
        "       const int ID0 = get_group_id(0)*TRANSPOSEX + tx; \n"
        "       const int ID1 = get_group_id(1)*TRANSPOSEY + ty; \n"
        "       __local float buffer[TRANSPOSEX][TRANSPOSEY];\n"
        "       if (ID0 < P && ID1 < Q) {\n"
        "             buffer[ty][tx] = input[ID1*P + ID0];\n"
        "       }\n"
        "       barrier(CLK_LOCAL_MEM_FENCE);\n"
        "       const int newID0 = get_group_id(1)*TRANSPOSEY + tx;\n"
        "       const int newID1 = get_group_id(0)*TRANSPOSEX + ty;\n"
        "       if (newID0 < Q && newID1 < P) {\n"
        "             output[newID1*Q + newID0] = buffer[tx][ty];\n"
        "       }\n"
        "}\n"

        "#define TS 1\n"
        "#define TSDK 1\n"
        "#define WPT 1\n"
        "#define RTS (TS/WPT)\n"
        "#define LPT ((TSDK*WPT)/(TS))\n"
        "kernel void matrixMul5(const int M, const int N, const int K, global const float *A, global const float *B, global float *C ){\n"
        "        const int row = get_local_id(0);\n"
        "        const int col = get_local_id(1); \n"
        "        const int globalRow = TS*get_group_id(0) + row; \n"
        "        const int globalCol = TS*get_group_id(1) + col;\n"
        "        __local float Asub[TSDK][TS];\n"
        "        __local float Bsub[TS][TSDK+2];\n"
        "        float acc[WPT];\n"
        "        for (int w=0; w<WPT; w++) \n"
        "                acc[w] = 0.0f;\n"
        "        const int numTiles = K/TSDK;\n"
        "        for (int t=0; t<numTiles; t++) {\n"
        "                for (int l=0; l<LPT; l++) {\n"
        "                        const int tiledIndex = TSDK*t + col + l*RTS;\n"
        "                        int indexA = (tiledIndex)*M + TS*get_group_id(0) + row;\n"
        "                        int indexB = (tiledIndex)*N + TS*get_group_id(1) + row;\n"
        "                        Asub[col + l*RTS][row] = A[indexA];\n"
        "                        Bsub[row][col + l*RTS] = B[indexB];\n"
        "                }\n"
        "                barrier(CLK_LOCAL_MEM_FENCE);\n"
        "                for (int k=0; k<TSDK; k++) {\n"
        "                       for (int w=0; w<WPT; w++) {\n"
        "                                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];\n"
        "                        }\n"
        "                }\n"
        "                barrier(CLK_LOCAL_MEM_FENCE);\n"
        "        }\n"
        "        for (int w=0; w<WPT; w++) {\n"
        "                C[(globalCol + w*RTS)*M + globalRow] = acc[w];\n"
        "        }\n"
        "}\n";



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
