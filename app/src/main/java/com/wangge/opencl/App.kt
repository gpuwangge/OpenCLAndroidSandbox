package com.wangge.opencl

import android.app.Application
import com.wangge.opencl.jni.OpenCL4J

class App: Application() {

    override fun onCreate() {
        super.onCreate()
        OpenCL4J.load()
    }

}