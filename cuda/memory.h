#ifndef CUDA_MEMORY_H
#define CUDA_MEMORY_H

//#include "./hallo-world.cuh"

template <typename T> T * memcopy(
	T * p_dst, T const * const p_src,
	std::size_t const count,
	cudaMemcpyKind const kind) {
	if ((p_dst != nullptr) && (p_src != nullptr) && (count > 0)) {
		pfc::check(cudaMemcpy(p_dst, p_src, count * sizeof(T), kind));
	}

	return p_dst;
}

template <typename T> T* & free_on_device(T* & dp_memory) {
	if (dp_memory != nullptr) {
		pfc::check(cudaFree(dp_memory)); dp_memory = nullptr;
	}

	return dp_memory;
}

template <typename T> T* malloc_on_device(std::size_t const count) {
	T * dp_memory = nullptr;

	if (count > 0) {
		pfc::check(cudaMalloc(&dp_memory, count * sizeof(T)));
	}

	return dp_memory;
}

/*
void copy_string(char * const hp_dst, char const * const hp_src,
	std::size_t count,
	std::size_t threads_in_block) {

	auto * dp_src = malloc_on_device<char>(count);
	auto * dp_dst = malloc_on_device<char>(count);

	memcopy(dp_src, hp_src, count, cudaMemcpyHostToDevice);

	std::size_t const blocks_in_grid = (count + threads_in_block - 1) / threads_in_block;

//	copy_string_kernel_wrapper(threads_in_block, blocks_in_grid, dp_dst, dp_src, count);

	pfc::check(cudaGetLastError());

	memcopy(hp_dst, dp_dst, count, cudaMemcpyDeviceToHost);

	free_on_device(dp_src);
	free_on_device(dp_dst);

	pfc::check(cudaDeviceSynchronize());
}
*/

#endif
