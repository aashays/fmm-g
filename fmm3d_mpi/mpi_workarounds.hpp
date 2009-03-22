#ifndef MPI_WORKAROUND_HEADER
#define MPI_WORKAROUND_HEADER

#include "mpi.h"

int MPI_Alltoallv_viaSends ( 
    void *sendbuf, 
    int *sendcnts, 
    int *sdispls, 
    MPI_Datatype sendtype, 
    void *recvbuf, 
    int *recvcnts, 
    int *rdispls, 
    MPI_Datatype recvtype, 
    MPI_Comm comm );

#endif

