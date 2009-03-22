#ifndef MANAGE_PETSC_EVENTS_HEADER
#define MANAGE_PETSC_EVENTS_HEADER

#include "petsc.h"

#if (PETSC_VERSION_RELEASE == 0)  // if using development version
extern PetscLogEvent  let_setup_event;
extern PetscLogEvent  fmm_setup_event;
extern PetscLogEvent  fmm_eval_event;
extern PetscLogEvent  fmm_check_event;
extern PetscLogEvent  a2a_numOct_event;
extern PetscLogEvent  a2aV_octData_event;
extern PetscLogEvent  EvalIni_event;
extern PetscLogEvent  EvalCtb2GlbExa_event;
extern PetscLogEvent  EvalGlb2UsrExaBeg_event;
extern PetscLogEvent  EvalUpwComp_event;
extern PetscLogEvent  EvalCtb2GlbEqu_event;
extern PetscLogEvent  EvalGlb2UsrEquBeg_event;
extern PetscLogEvent  EvalGlb2UsrExaEnd_event;
extern PetscLogEvent  EvalUList_event;
extern PetscLogEvent  EvalGlb2UsrEquEnd_event;
extern PetscLogEvent  EvalVList_event;
extern PetscLogEvent  EvalWList_event;
extern PetscLogEvent  EvalXList_event;
extern PetscLogEvent  EvalCombine_event;
extern PetscLogEvent  EvalFinalize_event;
extern PetscLogEvent  Ctb2GlbSctCreate_event; 
extern PetscLogEvent  Usr2GlbSctCreate_event; 
extern PetscLogStage  stages[6];
#else  // if using release version
extern PetscEvent  let_setup_event;
extern PetscEvent  fmm_setup_event;
extern PetscEvent  fmm_eval_event;
extern PetscEvent  fmm_check_event;
extern PetscEvent  a2a_numOct_event;
extern PetscEvent  a2aV_octData_event;
extern PetscEvent  EvalIni_event;
extern PetscEvent  EvalCtb2GlbExa_event;
extern PetscEvent  EvalGlb2UsrExaBeg_event;
extern PetscEvent  EvalUpwComp_event;
extern PetscEvent  EvalCtb2GlbEqu_event;
extern PetscEvent  EvalGlb2UsrEquBeg_event;
extern PetscEvent  EvalGlb2UsrExaEnd_event;
extern PetscEvent  EvalUList_event;
extern PetscEvent  EvalGlb2UsrEquEnd_event;
extern PetscEvent  EvalVList_event;
extern PetscEvent  EvalWList_event;
extern PetscEvent  EvalXList_event;
extern PetscEvent  EvalCombine_event;
extern PetscEvent  EvalFinalize_event;
extern PetscEvent  Ctb2GlbSctCreate_event; 
extern PetscEvent  Usr2GlbSctCreate_event; 
extern int  stages[6];
#endif

void registerPetscEvents();
PetscErrorCode DumpPerCoreSummary(MPI_Comm comm, const char * fname);

#endif
