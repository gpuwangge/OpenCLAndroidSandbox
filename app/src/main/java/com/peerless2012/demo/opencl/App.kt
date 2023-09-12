package com.peerless2012.demo.opencl

import android.app.Application
import com.peerless2012.demo.opencl.jni.OpenCL4J

class App: Application() {

    override fun onCreate() {
        super.onCreate()
        OpenCL4J.load()
    }

}