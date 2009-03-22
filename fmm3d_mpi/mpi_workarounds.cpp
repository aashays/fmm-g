#include "mpi_workarounds.hpp"

int MPI_Alltoallv_viaSends ( 
    void *_sendbuf, 
    int *sendcnts, 
    int *sdispls, 
    MPI_Datatype sendtype, 
    void *_recvbuf, 
    int *recvcnts, 
    int *rdispls, 
    MPI_Datatype recvtype, 
    MPI_Comm comm )
{
  int npes, rank;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &rank);

  int commCnt = 0;

  for(int i = 0; i < npes; i++) {
    if(sendcnts[i] > 0) {
      commCnt++;
    }
    if(recvcnts[i] > 0) {
      commCnt++;
    }
  }

  MPI_Request* requests = new MPI_Request[commCnt];
  MPI_Status* statuses = new MPI_Status[commCnt];

  commCnt = 0;

  //First place all recv requests. 
  char * recvbuf = (char*)_recvbuf;
  MPI_Aint recvtype_extent;
  MPI_Type_extent( recvtype, &recvtype_extent); 

  for(int i = 0; i < npes; i++) {
    if(recvcnts[i] > 0) {
      MPI_Irecv(
	  recvbuf+recvtype_extent*rdispls[i], 
	  recvcnts[i],
	  recvtype, 
	  i, 
	  1,
	  comm, 
	  requests+commCnt );
      commCnt++;
    }
  }

  //Next send the messages.
  char * sendbuf = (char*)_sendbuf;
  MPI_Aint sendtype_extent;
  MPI_Type_extent( sendtype, &sendtype_extent); 

  for(int i = 0; i < npes; i++) {
    if(sendcnts[i] > 0) {
      MPI_Isend( 
	  sendbuf+sendtype_extent*sdispls[i], 
	  sendcnts[i], 
	  sendtype, 
	  i, 
	  1,
	  comm, 
	  requests+commCnt );
      commCnt++;
    }
  }

  MPI_Waitall(commCnt, requests, statuses);
  delete [] requests;
  delete [] statuses;
  return MPI_SUCCESS;
}

