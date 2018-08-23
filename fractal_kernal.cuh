#if !defined FRACTAL_KERNAL_CUH
#define      FRACTAL_KERNAL_CUH

#include <string>
#include <functional>
#include "fractal.cuh"
#include "cuda/exception.h"
#include "./complex.cuh"
#include "./bitmap.cuh"
#include "./config.cuh"


	// colortable in const memory
	__constant__ RGB_T dp_cst_color_table[MAXITERATIONS + 1];

	__global__ void generateCuda_kernel(
		RGB_T * dp_result_image,
		RGB_T * dp_color_table,
		COMPLEX_T const real_stepsize,
		COMPLEX_T const imag_stepsize,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::size_t const img_width,
		std::size_t const img_heigth,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val)
	{ 	
		if ((dp_result_image != nullptr) && (dp_color_table != nullptr)) {

			auto const x = blockIdx.x * blockDim.x + threadIdx.x;
			auto const y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < img_width && y < img_heigth) {
				pfc::complex<COMPLEX_T> val(
					window_top_left.real() + (x * real_stepsize), 
					windows_bottom_right.imag() + (y * imag_stepsize)
				);

				int iter = fractal::calcJuliaVal(val, c, iter_max, dist_max);

				dp_result_image[y * img_width + x] = dp_color_table[iter];
			}
		}
	}

	__global__ void generateCuda_const_kernel(
		RGB_T * dp_result_image,
		COMPLEX_T const real_stepsize,
		COMPLEX_T const imag_stepsize,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::size_t const img_width,
		std::size_t const img_heigth,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val)
	{
		if ((dp_result_image != nullptr) && (dp_cst_color_table != nullptr)) {

			auto const x = blockIdx.x * blockDim.x + threadIdx.x;
			auto const y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x < img_width && y < img_heigth) {
				pfc::complex<COMPLEX_T> val(
					window_top_left.real() + (x * real_stepsize),
					windows_bottom_right.imag() + (y * imag_stepsize)
				);

				int iter = fractal::calcJuliaVal(val, c, iter_max, dist_max);

				dp_result_image[y * img_width + x] = dp_cst_color_table[iter];
			}
		}
	}

	__global__ void generateCuda_shared_kernel(
		RGB_T * dp_result_image,
		COMPLEX_T const real_stepsize,
		COMPLEX_T const imag_stepsize,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::size_t const img_width,
		std::size_t const img_heigth,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val)
	{
		__shared__ pfc::RGB_3_t dp_shared_color_table[MAXITERATIONS + 1];

		if ((dp_result_image != nullptr) && (dp_cst_color_table != nullptr)) {

			auto const x = blockIdx.x * blockDim.x + threadIdx.x;
			auto const y = blockIdx.y * blockDim.y + threadIdx.y;
			auto const idx = threadIdx.x * threadIdx.y;

			// copy color table to shared memory
			if (idx < (MAXITERATIONS + 1)) {
				dp_shared_color_table[idx] = dp_cst_color_table[idx];
			}

			__syncthreads();   // wait for all threads to finish loading



			if (x < img_width && y < img_heigth) {
				pfc::complex<COMPLEX_T> val(
					window_top_left.real() + (x * real_stepsize),
					windows_bottom_right.imag() + (y * imag_stepsize)
				);

				int iter = fractal::calcJuliaVal(val, c, iter_max, dist_max);

				dp_result_image[y * img_width + x] = dp_shared_color_table[iter];
			}
		}
	}

	__host__ void generateCuda_kernel_wrapper(
		dim3 const threads_in_block,
		dim3 const blocks_in_grid,
		RGB_T * dp_result_image,
		RGB_T * dp_color_table,
		double const real_stepsize,
		double const imag_stepsize,
		pfc::complex<COMPLEX_T> window_top_left,
		pfc::complex<COMPLEX_T> windows_bottom_right,
		std::size_t const img_width,
		std::size_t const img_heigth,
		pfc::complex<COMPLEX_T> c,
		size_t iter_max, COMPLEX_T dist_max,
		std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val, int optlevel)
	{
		if(optlevel == 0) {
			generateCuda_kernel <<<blocks_in_grid, threads_in_block>>> (
				dp_result_image, dp_color_table,
				real_stepsize, imag_stepsize, window_top_left, windows_bottom_right,
				img_width, img_heigth, c,
				iter_max, dist_max, calc_val
			);
		} else if (optlevel < 5) {
			generateCuda_const_kernel <<<blocks_in_grid, threads_in_block>>> (
				dp_result_image, real_stepsize, imag_stepsize,
				window_top_left, windows_bottom_right,
				img_width, img_heigth, c,
				iter_max, dist_max, calc_val
			);
		} else if (optlevel < 7) {
			generateCuda_shared_kernel <<<blocks_in_grid, threads_in_block>>> (
				dp_result_image, real_stepsize, imag_stepsize,
				window_top_left, windows_bottom_right,
				img_width, img_heigth, c,
				iter_max, dist_max, calc_val
			);
		} else {
			generateCuda_kernel <<<blocks_in_grid, threads_in_block>>> (
				dp_result_image, dp_color_table,
				real_stepsize, imag_stepsize, window_top_left, windows_bottom_right,
				img_width, img_heigth, c,
				iter_max, dist_max, calc_val
			);
		}
	}

#endif // FRACTAL_KERNAL_CUH
