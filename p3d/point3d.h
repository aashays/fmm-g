/**
 * \file src/point3d.h
 * \brief Implements a structure for u-list computation
 */

#if !defined (INC_POINT3D_H)
#define INC_POINT3D_H /*!< point3d.h included */

#if defined (__cplusplus)
extern "C" {
#endif

/* ============================================================================
 */
//#define DS_ORG
/* \brief point3d structure */
typedef struct point3d {

/*	P->src_ has the form x1 y1 z1 d1 x2 y2 z2 d2 ......
	P->trg has the form x1 y1 z1 x2 y2 z2
	P->trgVal has the form t1 t2 t3.....*/

  int numSrc;     /* number of source points */
  int numTrg;     /* number of target points */
  int numTrgBox;  /* number of target boxes */
  int numSrcBox;  /* number of source boxes */
  int dim;	  /* dimension */

#ifdef DS_ORG
  float* src_;    /* source coordinates */
  float* trg_;    /* target coordinates */
#else
  float* sx_;    /* source x coordinates */
  float* sy_;    /* source y coordinates */
  float* sz_;    /* source z coordinates */

  float* tx_;    /* target x coordinates */
  float* ty_;    /* target y coordinates */
  float* tz_;    /* target z coordinates */
#endif
  float* srcDen; /* source density values of size numSrc */
  float* trgVal; /* target values of size numTrg */
  float *trgValC;	//TODO: remove

  int** uList; /* u-list for each target box which is a leaf node */
  int* uListLen; /* u-list length for each target box */

  int* srcBoxSize; /* number of points in source boxes */
  int* trgBoxSize; /* number of points in target boxes */

} point3d_t;

#if defined (__cplusplus)
}
#endif
#endif

/* eof */
