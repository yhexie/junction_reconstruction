#ifndef PCL_WRAPPER_EXPORTS_H_
#define PCL_WRAPPER_EXPORTS_H_

#if defined WIN32 || defined _WIN32 || defined WINCE || defined __MINGW32__
    #ifdef PCL_WRAPPER_API_EXPORTS
        #define PCL_WRAPPER_EXPORTS __declspec(dllexport)
    #else
        #define PCL_WRAPPER_EXPORTS __declspec(dllimport)
    #endif
#else
    #define PCL_WRAPPER_EXPORTS
#endif

#endif  //#ifndef PCL_WRAPPER_EXPORTS_H_
