package com.wangge.opencl.activity

//import android.opengl.GLSurfaceView
import android.os.Bundle
//import android.view.Menu
import androidx.appcompat.app.AppCompatActivity
//import androidx.appcompat.widget.Toolbar
import com.wangge.opencl.jni.uBenchmarkManager

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //setContentView(R.layout.activity_main)
        //val toolbar = findViewById<Toolbar>(R.id.toolbar)
        //setSupportActionBar(toolbar)

        var bSucceed = false
        bSucceed = uBenchmarkManager.runVectorAdd(100)
        bSucceed = uBenchmarkManager.runMatrixMul1(256, 32)  //Native(slowest)
        bSucceed = uBenchmarkManager.runMatrixMul5(256,32,8,16,16)  //Transpose
        bSucceed = uBenchmarkManager.runMatrixMul6(256,128,128,8,8,16,16)  //Register(fastest)
    }

    //override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
    //    menuInflater.inflate(R.menu.main, menu)
    //    return true
    //}

}