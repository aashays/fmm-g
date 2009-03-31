#ifndef GPU_SETUP_H_
#define GPU_SETUP_H_

#include <cstddef>
#include <p3d/point3d.h>
#include <p3d/upComp.h>
#include <p3d/dnComp.h>

#define GPU_CERR /*!< Enables verbose GPU error messages */

#if defined(__cplusplus)
extern "C" {
#endif

  /** Returns the number of available GPU devices. */
  size_t gpu_count (void);

  /**
   *  Dumps information about a GPU device to stderr, where 0 <= dev_id
   *  < gpu_count()
   */
  void gpu_dumpinfo (size_t dev_id);

  /** \name Prints GPU-related messages, including MPI rank & GPU device information. */
  /*@{*/
  void gpu_msg__stdout (const char* msg, const char* filename, size_t lineno);
# define GPU_MSG(msg)  gpu_msg__stdout(msg, __FILE__, __LINE__)
  /*@}*/
  
  /** Selects a GPU by ID, 0 <= dev_id < gpu_count() */
  void gpu_select (size_t dev_id);
  
  /** Allocates a block of 'n' floats on the GPU, and initializes them to zero. */
  float* gpu_calloc_float (size_t n);

  /** Allocates a block of 'n' ints on the GPU, and initializes them to zero. */
  int* gpu_calloc_int (size_t n);
  
  /** Copies from host memory to device memory */
  void gpu_copy_cpu2gpu_float (float* d, const float* s, size_t n);
  
  /** Copies from host memory to device memory */
  void gpu_copy_cpu2gpu_int (int* d, const int* s, size_t n);
  
  /** Copies from device memory to host memory */
  void gpu_copy_gpu2cpu_float (float* d, const float* s, size_t n);
  
  /** Performs U-list computation */
  void dense_inter_gpu (point3d_t*);
  
  void gpu_up(upComp_t*);

  void gpu_down(dnComp_t*);

#if defined (GPU_CERR)
  /** Dumps an error if the last GPU call failed. */
  void gpu_checkerr__stdout (const char* filename, size_t line);
  /** Dumps an error if the last GPU call failed. */
#  define GPU_CE  gpu_checkerr__stdout (__FILE__, __LINE__)
#else /* ! defined (GPU_CERR) */
  /** No op */
#  define GPU_CE
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif	//GPU_SETUP_H_
