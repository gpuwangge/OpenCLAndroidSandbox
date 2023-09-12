package com.integratekhronosgroup.opencl.jni

import android.util.Log

object OpenCLJNI {

    const val TAG = "OpenCL"

    var findOpenCL = false

    init {
        try {
            // /vendor/lib64/libOpenCL.so
            System.loadLibrary("OpenCL")
            Log.i(TAG, "OpenCL load success!!! (package com.integratekhronosgroup.opencl.jni)")
            findOpenCL = true
        } catch (exception: UnsatisfiedLinkError) {
            exception.printStackTrace()
            //try {
            //    System.load("/vendor/lib64/libOpenCL.so")
            //    Log.i(TAG, "OpenCL load success!!! second try")
            //    findOpenCL = true
            //} catch (e2: UnsatisfiedLinkError) {
            //    Log.e(TAG, "OpenCL load Fail!!!")
            //    e2.printStackTrace()
            //}
        }
    }

}