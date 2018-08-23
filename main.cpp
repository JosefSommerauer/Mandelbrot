#include <iostream>
#include <array>
#include <string>
#include <cstring>
 
#include <chrono>
#include <ctime>

#include <system_error>
#include <cuda_runtime.h>

#include "./cuda/exception.h"

#include "fractal.cuh"
#include "color_mapper.h"
#include "timer.h"
#include "./cuda/cuda_timer.h"

#include "./config.cuh"

// benchmark
void benchmark_gpu (fractal & frac, dim3 threads_in_block, int optlevel)
{
   std::chrono::time_point<std::chrono::system_clock> start, end;
	pfc::cuda_timer cuda_timer;

	// start timer
	start = std::chrono::system_clock::now();		
	cuda_timer.start();

	auto cpu_timer = pfc::timed_run(
			&fractal::generateJuliaSlowestFastestCuda,
			&frac, threads_in_block, optlevel
	);

	// stop timer
	end = std::chrono::system_clock::now();		
	int sec_cnt = std::chrono::duration_cast<std::chrono::microseconds>
		                 (end-start).count();
		
	// print result
	std::cout << threads_in_block.x << ";"
			  << threads_in_block.y << ";"
			  << cuda_timer.stop() << ";"
			  << cpu_timer.count() << ";"
              << sec_cnt << ";" << std::endl;
}

void benchmark_cpu (fractal & frac, size_t num_threads) 
{
   std::chrono::time_point<std::chrono::system_clock> start, end;
	pfc::cuda_timer cuda_timer;

	// start timer
	start = std::chrono::system_clock::now();		
	cuda_timer.start();

	auto cpu_timer = pfc::timed_run(
			&fractal::generateJuliaCarpetMultithreated, 
			&frac, num_threads
	);

	// stop timer
	end = std::chrono::system_clock::now();		
	int sec_cnt = std::chrono::duration_cast<std::chrono::microseconds>
		                 (end-start).count();
		
	// print result
	std::cout << num_threads << ";"
			  << cuda_timer.stop() << ";"
			  << cpu_timer.count() << ";"
              << sec_cnt << ";" << std::endl;
}

int main() {
	try {
		pfc::check(cudaGetLastError());
		cudaDeviceReset(); // reset device

		size_t img_size = IMAGESIZE;
		const size_t iter_max = MAXITERATIONS;
		const double dist_max = 2.0;

		fractal mFractal(img_size, img_size, iter_max, dist_max);

		// generate color table
		mFractal.gernerateColorTable(
			pfc::get_rgb_piecewise_linear<RGB_T>
		);

		//test cpu
#ifdef TEST_CPU
		std::cout << "generate fractal on cpu" << std::endl;	
		std::cout << "threads;" 
			 << "gpu_ticks;" << "cpu_ticks;"
          << "time_span;" << std::endl;

		for(int i=1;i<=1024;i*=2) {
			benchmark_cpu(mFractal, i);
		}

		std::cout << std::endl << std::endl;
#endif

		// test gpu
		std::cout << "generate fractal on gpu with optimisation level:" << OPTIMISATION_LEVEL << std::endl;
		std::cout << "threads_in_block_x;" << "threads_in_block_y;"
			 << "gpu_ticks;" << "cpu_ticks;" 
			 << "time_span;" << std::endl << std::flush;

#if OPTIMISATION_LEVEL >= 4
		benchmark_gpu(mFractal, dim3(32,32), OPTIMISATION_LEVEL);
#else
		benchmark_gpu(mFractal, dim3(16,16), OPTIMISATION_LEVEL);
#endif

	} catch (pfc::cuda_exception& e) {
		std::cout << "CUDA EXCEPTION: " << e.what() << std::endl;
    } catch(const std::system_error& e) {
        std::cout << "Caught system_error with code " << e.code() 
                  << " meaning " << e.what() << '\n';
    }

	cudaDeviceReset();
}
