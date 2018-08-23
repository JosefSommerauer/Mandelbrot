#include <thread>
#include <vector>
#include <sstream>

#include "fractal.cuh"
#include "complex.cuh"
#include "./color_mapper.h"

#include "./cuda/exception.h"
#include "./cuda/memory.h"
#include "./fractal_kernal.cuh"

#include "./config.cuh"

fractal::fractal(size_t image_width, size_t image_heigth,
	size_t iter_max, COMPLEX_T dist_max)
	: mImage(0), mIterMax(iter_max), mDistMax(dist_max), dp_color_table(0)
{
#if OPTIMISATION_LEVEL < 7
	mImage = new pfc::bitmap(image_width, image_heigth);
	mColorTable = new RGB_T[mIterMax +1];
#else
	cudaSetDeviceFlags(cudaDeviceMapHost); // allow zero copy
	pfc::check(cudaGetLastError());
	mImage = new pfc::bitmap(image_width, image_heigth);
	pfc::check(cudaHostAlloc((void **)&mColorTable,  sizeof(RGB_T)*(mIterMax +1),  cudaHostAllocMapped));
#endif
}

fractal::~fractal()
{
#if OPTIMISATION_LEVEL < 7
	delete[] mColorTable;
#else
	cudaFreeHost(mColorTable);
#endif

#if OPTIMISATION_LEVEL == 0
	free_on_device(dp_color_table);
#endif
}

void fractal::gernerateColorTable(
		std::function<RGB_T(const int, size_t)> color_mapper)
{


	for(int i=0; i< mIterMax +1;i++) {
		mColorTable[i] = color_mapper(i, mIterMax);
	}

#if OPTIMISATION_LEVEL == 0
	dp_color_table = malloc_on_device<RGB_T>(mIterMax + 1);
	memcopy(dp_color_table, mColorTable, mIterMax + 1, cudaMemcpyHostToDevice);
#elif OPTIMISATION_LEVEL <= 6 // use const memory
	// copy to const mem.
	pfc::check(cudaMemcpyToSymbol(dp_cst_color_table,
		mColorTable, sizeof(RGB_T)*(mIterMax + 1)));
#else
	cudaHostGetDevicePointer((void **)&dp_color_table,  (void *)mColorTable , 0);
	pfc::check(cudaGetLastError());
#endif

}

void fractal::generate(pfc::complex<COMPLEX_T> c,
	pfc::complex<COMPLEX_T> window_top_left,
	pfc::complex<COMPLEX_T> windows_bottom_right,
	std::function<size_t(pfc::complex<COMPLEX_T>,
	pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val,
	size_t y_from, size_t y_to, size_t x_form, size_t x_to)
{
	if (y_to == 0) y_to = mImage->get_height();
	if (x_to == 0) x_to = mImage->get_width();

	COMPLEX_T real_stepsize = (-1.0*(window_top_left.real())
			+ windows_bottom_right.real()) / mImage->get_width();
	COMPLEX_T imag_stepsize = (-1.0*(windows_bottom_right.imag())
			+ window_top_left.imag()) / mImage->get_height();
	COMPLEX_T real = window_top_left.real() + (x_form*real_stepsize);
	COMPLEX_T imag = windows_bottom_right.imag() + (y_from*imag_stepsize);

	for (size_t y = y_from; y < y_to; y++) {
		for (size_t x = x_form; x < x_to; x++)	{

			pfc::complex<COMPLEX_T> val(real, imag);
			int iter = calc_val(val, c, mIterMax, mDistMax);
			mImage->get_image_RGB_3()[y * mImage->get_width() + x] = mColorTable[iter];
			real += real_stepsize;
		}

		real = window_top_left.real() + (x_form*real_stepsize);
		imag += imag_stepsize;
	}
}

void fractal::generateMT(pfc::complex<COMPLEX_T> c,
	pfc::complex<COMPLEX_T> window_top_left,
	pfc::complex<COMPLEX_T> windows_bottom_right,
	std::function<size_t(pfc::complex<COMPLEX_T>,
	pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val,
	size_t number_of_threads) {

	if (number_of_threads == 0) number_of_threads = 1;

	std::vector<std::thread> t;

	size_t lines_per_thread = mImage->get_height() / number_of_threads;

	for (size_t i = 0; i < number_of_threads -1; i++) {
		t.push_back(std::thread(&fractal::generate, this, c,
			window_top_left, windows_bottom_right,
			calc_val, i*lines_per_thread,
			(i + 1)*lines_per_thread, 0, 0));
	}

	// calc rest
	t.push_back(std::thread(&fractal::generate, this, c,
		window_top_left, windows_bottom_right,
		calc_val, (number_of_threads-1)*lines_per_thread,
		mImage->get_height(), 0, 0));

	// wait for threads to finish
	for (size_t i = 0; i < t.size(); i++) {
		t[i].join();
	}

}

// gernerate with cuda
void fractal::generateCuda(pfc::complex<COMPLEX_T> c,
	pfc::complex<COMPLEX_T> window_top_left,
	pfc::complex<COMPLEX_T> windows_bottom_right,
	std::function<size_t(pfc::complex<COMPLEX_T>, pfc::complex<COMPLEX_T>, size_t, COMPLEX_T)> calc_val,
	dim3 threads_in_block, int optlevel) {

		int count = 0;
		pfc::check(cudaGetDeviceCount(&count));

		if (count > 0) {
			pfc::check(cudaSetDevice(0));

			// allocate memory on device
			size_t ImagePixels = mImage->get_width() * mImage->get_height();

		#if OPTIMISATION_LEVEL < 7
			RGB_T * dp_result_image = malloc_on_device<RGB_T>(ImagePixels);
		#else
			RGB_T * dp_result_image = 0;
			cudaHostGetDevicePointer((void **)&dp_result_image, (void *)mImage->get_image_RGB_3() , 0);
			pfc::check(cudaGetLastError());
		#endif

			// start kernel funktion
			COMPLEX_T real_stepsize = (-1.0*(window_top_left.real())
				+ windows_bottom_right.real()) / mImage->get_width();
			COMPLEX_T imag_stepsize = (-1.0*(windows_bottom_right.imag())
				+ window_top_left.imag()) / mImage->get_height();

			dim3 blocks_in_grid(
				mImage->get_width() / threads_in_block.x,
				mImage->get_height() / threads_in_block.y
			);

			generateCuda_kernel_wrapper(
				threads_in_block,	blocks_in_grid,
				dp_result_image, dp_color_table,
				real_stepsize, imag_stepsize,
				window_top_left, windows_bottom_right,
				mImage->get_width(), mImage->get_height(),
				c,	mIterMax, mDistMax, calc_val, optlevel
			);


			pfc::check(cudaDeviceSynchronize());
			pfc::check(cudaGetLastError());

		#if OPTIMISATION_LEVEL < 7
			// copy result to host
			memcopy(mImage->get_image_RGB_3(), dp_result_image, ImagePixels, cudaMemcpyDeviceToHost);
			// free mem on device
			free_on_device(dp_result_image);
		#endif

		} else {
			std::cout << "no cuda device found!" << std::endl;
		}
}

// gernerate carpet with cuda
void fractal::generateJuliaCarpetCuda(dim3 threads_in_block) {
	auto top_left = pfc::complex<COMPLEX_T>(-1.5, 1.5);
	auto bottom_right = pfc::complex<COMPLEX_T>(1.5, -1.5);
	auto calc_val = fractal::calcJuliaVal;

	for(COMPLEX_T x=-1.0;x<=1.0;x+=0.5) {
		for(COMPLEX_T y=-1.0;y<=1.0;y+=0.5) {
			auto c = pfc::complex<COMPLEX_T>(x, y);

			generateCuda(c, top_left, bottom_right,
				calc_val, threads_in_block, 0);

			std::ostringstream strs;
			strs << "julia_" << x << "_" << y << ".bmp";
			save(strs.str());
		}
	}
}

void fractal::generateJuliaSlowestFastestCuda(dim3 threads_in_block, int optlevel) {
	auto top_left = pfc::complex<COMPLEX_T>(-1.5, 1.5);
	auto bottom_right = pfc::complex<COMPLEX_T>(1.5, -1.5);
	auto calc_val = fractal::calcJuliaVal;

	// slowest
	auto c = pfc::complex<COMPLEX_T>(0, 0);
	generateCuda(c, top_left, bottom_right,
		calc_val, threads_in_block, optlevel);

	save("julia1.bmp");

	// fastest
	c = pfc::complex<COMPLEX_T>(-1.0, -1.0);
	generateCuda(c, top_left, bottom_right,
		calc_val, threads_in_block, optlevel);

	save("julia2.bmp");
}

// gernerate carpet multithreaded
void fractal::generateJuliaCarpetMultithreated(size_t number_of_threads) {
	auto top_left = pfc::complex<COMPLEX_T>(-1.5, 1.5);
	auto bottom_right = pfc::complex<COMPLEX_T>(1.5, -1.5);
	auto calc_val = fractal::calcJuliaVal;

	for(COMPLEX_T x=-1.0;x<=1.0;x+=0.5) {
		for(COMPLEX_T y=-1.0;y<=1.0;y+=0.5) {
			auto c = pfc::complex<COMPLEX_T>(x, y);

			generateMT(c, top_left, bottom_right,
				calc_val, number_of_threads);
		}
	}
}

