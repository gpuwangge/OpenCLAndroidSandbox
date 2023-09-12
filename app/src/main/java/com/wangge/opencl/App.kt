package com.wangge.opencl

import android.app.Application
import com.wangge.opencl.jni.uBenchmarks

class App: Application() {

    override fun onCreate() {
        super.onCreate()
        uBenchmarks.load()
    }

}