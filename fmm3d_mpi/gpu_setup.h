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

extern "C" void dense_inter_gpu (point3d_t*);

#endif	//GPU_SETUP_H_
