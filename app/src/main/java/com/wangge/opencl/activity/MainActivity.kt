package com.wangge.opencl.activity

//import android.opengl.GLSurfaceView
import android.os.Bundle
//import android.view.Menu
import androidx.appcompat.app.AppCompatActivity
//import androidx.appcompat.widget.Toolbar
import com.wangge.opencl.jni.uBenchmarks

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //setContentView(R.layout.activity_main)

        //val toolbar = findViewById<Toolbar>(R.id.toolbar)

        //setSupportActionBar(toolbar)

        var bSucceed = false
        bSucceed = uBenchmarks.runVectorAdd()
        bSucceed = uBenchmarks.runMatrixMul1()
        bSucceed = uBenchmarks.runMatrixMul5()
        bSucceed = uBenchmarks.runMatrixMul6()
    }

    //override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
    //    menuInflater.inflate(R.menu.main, menu)
    //    return true
    //}

}