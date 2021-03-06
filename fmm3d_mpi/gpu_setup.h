#ifndef GPU_SETUP_H_
#define GPU_SETUP_H_

#include <cstddef>
#include <cstdio>
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
   *  Dumps information about a GPU device to 'fp', where 0 <= dev_id
   *  < gpu_count(). If 'fp' is NULL, then this routine creates a
   *  host-specific data file. Use the environment variable 'LOG_DIR'
   *  to have the data placed into a specific output directory.
   */
  void gpu_dumpinfo (FILE* fp, size_t dev_id);

  /** \name Prints GPU-related messages, including MPI rank & GPU device information. */
  /*@{*/
  void gpu_msg__stdout (const char* msg, const char* filename, size_t lineno);
# define GPU_MSG(msg)  gpu_msg__stdout(msg, __FILE__, __LINE__)
  /*@}*/
  
  /** Selects a GPU by ID, 0 <= dev_id < gpu_count() */
  void gpu_select (size_t dev_id);
  
  /** Allocates a block of 'n' bytes on the GPU, and initializes them to zero. */
  void* gpu_calloc (size_t n);

  /** Allocates a block of 'n' floats on the GPU, and initializes them to zero. */
  float* gpu_calloc_float (size_t n);

  /** Allocates a block of 'n' ints on the GPU, and initializes them to zero. */
  int* gpu_calloc_int (size_t n);
  
  /** \name Copies between host and device (GPU) memory. */
  /*@{*/
  void gpu_copy_cpu2gpu (void* d, const void* s, size_t n);
  void gpu_copy_cpu2gpu_float (float* d, const float* s, size_t n);
  void gpu_copy_cpu2gpu_int (int* d, const int* s, size_t n);
  void gpu_copy_gpu2cpu (void* d, const void* s, size_t n);
  void gpu_copy_gpu2cpu_float (float* d, const float* s, size_t n);
  /*@}*/

  /** Checks that a pointer is non-NULL, and aborts with an error if it is. */
  void gpu_check_pointer (const void* p, const char* filename, size_t line);
  
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

  /** Checks for a non-NULL pointer */
#  define GPU_CP(p)  gpu_check_pointer ((p), __FILE__, __LINE__)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif	//GPU_SETUP_H_
