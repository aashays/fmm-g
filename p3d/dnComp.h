/**
 * \file src/dnComp.h
 * \brief Implements a structure for downward computation
 */

#if !defined (INC_DNCOMP_H)
#define INC_DNCOMP_H /*!< dnComp.h included */

/* ============================================================================
 */
/* \brief dnComp structure */
typedef struct dnComp {

/*	DnC->trg_ has the form x1 y1 z1 x2 y2 z2 ......
	DnC->trgVal has the form t1 t2 t3.....*/

  int tag;
  int numTrg;     /* number of target points */
  int numTrgBox;  /* number of target boxes */
  int dim;	  /* dimension */

#ifdef DS_ORG
  float* trg_;    /* target coordinates */
#endif
  float** trgVal; /* target potentials */
  int srcDim;	//152 for 6, 56 for 4, forget about 8

  int* trgBoxSize; /* number of points in target boxes */
  float* srcCtr;   /* center of the source box */
  float* srcRad;   /* radius of the source box */
//  FltNumMat samPos; /* sample position */
  float* srcDen;

  float *samPosF;	//pointer to sampos array

} dnComp_t;

#endif

/* eof */
