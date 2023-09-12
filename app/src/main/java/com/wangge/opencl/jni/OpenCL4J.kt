package com.wangge.opencl.jni

import com.integratekhronosgroup.opencl.jni.OpenCLJNI

object OpenCL4J {
    init {
        if (OpenCLJNI.findOpenCL)
            System.loadLibrary("OpenCL4J")
    }
    fun load() {}

    fun wangge(): Boolean{
        if (!OpenCLJNI.findOpenCL) return false
        //return vectorAdd()
        return matrixMul()
    }

    private external fun vectorAdd(): Boolean
    private external fun matrixMul(): Boolean


}