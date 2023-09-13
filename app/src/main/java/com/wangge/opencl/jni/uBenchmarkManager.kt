package com.wangge.opencl.jni

import com.integratekhronosgroup.opencl.jni.OpenCLJNI

object uBenchmarkManager {
    init {
        if (OpenCLJNI.findOpenCL)
            System.loadLibrary("OpenCLuBenchmarks")
    }
    fun load() {}

    fun runVectorAdd(maxNDRange: Int): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return vectorAdd(maxNDRange)
    }
    fun runMatrixMul1(DIM: Int, TILESIZE: Int): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return matrixMul1(DIM, TILESIZE)
    }
    fun runMatrixMul5(DIM: Int, TILESIZE: Int, WPT: Int, TRANSPOSEX: Int, TRANSPOSEY: Int): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return matrixMul5(DIM, TILESIZE, WPT, TRANSPOSEX, TRANSPOSEY)
    }
    fun runMatrixMul6(DIM: Int, TSM: Int, TSN: Int, WPTM: Int, WPTN: Int, TRANSPOSEX: Int, TRANSPOSEY: Int): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return matrixMul6(DIM, TSM, TSN, WPTM, WPTN, TRANSPOSEX, TRANSPOSEY)
    }

    private external fun vectorAdd(maxNDRange: Int): Boolean
    private external fun matrixMul1(DIM: Int, TILESIZE: Int): Boolean
    private external fun matrixMul5(DIM: Int, TILESIZE: Int, WPT: Int, TRANSPOSEX: Int, TRANSPOSEY: Int): Boolean
    private external fun matrixMul6(DIM: Int, TSM: Int, TSN: Int, WPTM: Int, WPTN: Int, TRANSPOSEX: Int, TRANSPOSEY: Int): Boolean
}