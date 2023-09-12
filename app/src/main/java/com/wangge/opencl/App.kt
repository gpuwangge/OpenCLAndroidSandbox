package com.wangge.opencl

import android.app.Application
import com.wangge.opencl.jni.uBenchmarkManager

class App: Application() {

    override fun onCreate() {
        super.onCreate()
        uBenchmarkManager.load()
    }

}