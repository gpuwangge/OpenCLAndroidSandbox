package com.wangge.opencl.activity

//import android.opengl.GLSurfaceView
import android.os.Bundle
//import android.view.Menu
import androidx.appcompat.app.AppCompatActivity
//import androidx.appcompat.widget.Toolbar
import com.wangge.opencl.jni.OpenCL4J

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        //setContentView(R.layout.activity_main)

        //val toolbar = findViewById<Toolbar>(R.id.toolbar)

        //setSupportActionBar(toolbar)

        var bwangge = OpenCL4J.wangge()
        //val glSurfaceView = findViewById<GLSurfaceView>(R.id.main_surface_view)
        //glSurfaceView.setEGLContextClientVersion(2)
        //glSurfaceView.setRenderer(GLInCLOutRender(applicationContext))
    }

    //override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
    //    menuInflater.inflate(R.menu.main, menu)
    //    return true
    //}

}