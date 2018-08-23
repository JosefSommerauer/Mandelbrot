#if !defined FRACTAL_CUH
#define      FRACTAL_CUH

#include <string>
#include <functional>
#include "./complex.cuh"
#include "./bitmap.cuh"
#include "./config.cuh"

class fractal {
public:
	fractal(size_t image_width, size_t image_heigth,
		size_t iter_max, COMPLEX_T dist_max);

	~fractal();

	// calculate color table
	void gernerateColorTable(std::function<RGB_T(const int, size_t)> color_mapper);

	// calculate single threaded
	void generate(pfc::complex<COMPLEX_T> c,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val,
		size_t y_from = 0, size_t y_to = 0, size_t x_form = 0, size_t x_to = 0);

	// gernerate multi threaded
	void generateMT(pfc::complex<COMPLEX_T> c,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val,
		size_t number_of_threads);

	// gernerate with cuda
	void generateCuda(pfc::complex<COMPLEX_T> c,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val,
		dim3 threads_in_block, int optlevel);

	// gernerate carpet with cuda
	void generateJuliaCarpetCuda(dim3 threads_in_block);

	// generate slowest and fastest julia set
	void generateJuliaSlowestFastestCuda(dim3 threads_in_block, int optlevel);

	// gernerate carpet multithreaded
	void generateJuliaCarpetMultithreated(size_t number_of_threads);

	// calculation functions 
	CUDA_ATTR_HOST_DEVICE inline static size_t calcJuliaVal(
		pfc::complex<COMPLEX_T> val,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max){
		size_t i = 0;

	#if OPTIMISATION_LEVEL == 6
		#pragma unroll
	#endif
		for (; i < iter_max && val.norm() < dist_max; i++) {
			val = val.square() + c;
		}

		return i;
	}

	CUDA_ATTR_HOST_DEVICE inline size_t calcJuliaPow5Val(
		pfc::complex<COMPLEX_T> val,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max) {
		size_t i = 0;

	#if OPTIMISATION_LEVEL == 6
		#pragma unroll
	#endif
		for (; i < iter_max && val.norm() < dist_max; i++) {
			val = val*val*val*val*val + c;
		}

		return i;
	}

	CUDA_ATTR_HOST_DEVICE inline size_t calcMandelVal(
		pfc::complex<COMPLEX_T> val,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max) {
		size_t i = 0;

	#if OPTIMISATION_LEVEL == 6
		#pragma unroll
	#endif
		for (; i < iter_max && val.norm() < dist_max; i++) {
			val = c.square() + val;
		}
		return i;
	}

	bool save(std::string filename) {
		return mImage->to_file(filename);
	}

private:
	pfc::bitmap * mImage;
	size_t mIterMax;
	COMPLEX_T mDistMax;
	
	RGB_T * mColorTable;

	// colortable in global memory
	RGB_T * dp_color_table;
};
#endif
