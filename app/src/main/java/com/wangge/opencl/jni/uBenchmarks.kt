package com.wangge.opencl.jni

import com.integratekhronosgroup.opencl.jni.OpenCLJNI

object uBenchmarks {
    init {
        if (OpenCLJNI.findOpenCL)
            System.loadLibrary("OpenCL4J")
    }
    fun load() {}

    fun runVectorAdd(): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return vectorAdd()
    }
    fun runMatrixMul1(): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return matrixMul1()
    }
    fun runMatrixMul5(): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        return matrixMul5()
    }

    private external fun vectorAdd(): Boolean
    private external fun matrixMul1(): Boolean
    private external fun matrixMul5(): Boolean

}