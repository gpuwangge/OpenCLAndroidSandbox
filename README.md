# OpenCLAndroidSandbox
![Android](https://img.shields.io/badge/Android-passing-brightgreen)

## Build
- Open folder with Android Studio  
- Sync  
- Build apk  

## Install
[path to adb.exe]./adb -s [device id] install [Path to .apk] 

## Troubleshooting
- After Rebuild/Clean project, apk build will fail because OpenCL functions are undefined symbol.   
Solution: Delete .gradle folder and build apk again.  

## Credits
- https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
- https://github.com/KhronosGroup/OpenCL-Headers.git
- https://github.com/peerless2012/AndroidOpenCL.git
