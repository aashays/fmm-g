#ifndef GPU_SETUP_H_
#define GPU_SETUP_H_

#include <cstddef>
#include <p3d/point3d.h>

/** Returns the number of available GPU devices. */
extern "C" size_t gpu_count (void);

/**
 *  Dumps information about a GPU device to stderr, where 0 <= dev_id
 *  < gpu_count()
 */
extern "C" void gpu_dumpinfo (size_t dev_id);

/** Selects a GPU by ID, 0 <= dev_id < gpu_count() */
extern "C" void gpu_select (size_t dev_id);

/** Allocates a block of 'n' floats on the GPU, and initializes them to zero. */
extern "C" float* gpu_calloc_float (size_t n);

/** Allocates a block of 'n' ints on the GPU, and initializes them to zero. */
extern "C" int* gpu_calloc_int (size_t n);

/** Copies from host memory to device memory */
extern "C" void gpu_copy_cpu2gpu_float (float* d, const float* s, size_t n);

/** Copies from host memory to device memory */
extern "C" void gpu_copy_cpu2gpu_int (int* d, const int* s, size_t n);

/** Copies from device memory to host memory */
extern "C" void gpu_copy_gpu2cpu_float (float* d, const float* s, size_t n);

/** Performs U-list computation */
extern "C" void dense_inter_gpu (point3d_t*);

#define GPU_CERR /*!< Enables verbose GPU error messages */
#if defined (GPU_CERR)
#  include <cstdio>
/** Dumps an error if the last GPU call failed. */
extern "C" void GPU_CE__ (FILE *, const char* filename, size_t line);
/** Dumps an error if the last GPU call failed. */
#  define GPU_CE(fp)  GPU_CE__ ((fp), __FILE__, __LINE__)
#else /* ! defined (GPU_CERR) */
/** No op */
#  define GPU_CE(fp)
#endif

#endif	//GPU_SETUP_H_
