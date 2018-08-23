#undef CUDA_ATTR_HOST_DEVICE

#if defined __CUDACC__   /* NVIDIA CUDA compiler */
#define CUDA_ATTR_HOST_DEVICE __device__ __host__
#else
#define CUDA_ATTR_HOST_DEVICE
#endif

#ifndef OPTIMISATION_LEVEL
	#define OPTIMISATION_LEVEL 8
#endif

#if OPTIMISATION_LEVEL >= 8
	#define IMAGESIZE 4096
#else
	#define IMAGESIZE 1000
#endif

#define MAXITERATIONS 500
//#define TEST_CPU

#if OPTIMISATION_LEVEL >= 3
	#define COMPLEX_T float
#else
	#define COMPLEX_T double
#endif

#if OPTIMISATION_LEVEL == 2
	#define RGB_T pfc::RGB_4_t
#else
	#define RGB_T pfc::RGB_3_t
#endif
