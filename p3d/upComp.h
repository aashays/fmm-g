/**
 * \file src/upComp.h
 * \brief Implements a structure for upward computation
 */

#if !defined (INC_UPCOMP_H)
#define INC_UPCOMP_H /*!< upComp.h included */

/* ============================================================================
 */
/* \brief upComp structure */


typedef struct upComp {

/*	P->src_ has the form x1 y1 z1 d1 x2 y2 z2 d2 ......
	P->trgVal has the form t1 t2 t3.....*/

  int tag;
  int numSrc;     /* number of source points */
  int numSrcBox;  /* number of source boxes */
  int dim;	  /* dimension */

#ifdef DS_ORG
  float* src_;    /* source coordinates */
#endif
  float* srcDen; /* source density values of size numSrc */
  float** trgVal; /* target potentials */	//why?????????
  int trgDim;	//296 for 6, 152 for 4, forget about 8

  int* srcBoxSize; /* number of points in source boxes */
  float* trgCtr;   /* center of the target box */
  float* trgRad;   /* radius of the target box */	//merge with ctr? float4
//#ifndef SAVE_ME_FROM_FLTNUMMAT
//  FltNumMat samPos; /* sample position */
//#endif
  float *samPosF;	//pointer to sampos array


} upComp_t;

#endif

/* eof */
