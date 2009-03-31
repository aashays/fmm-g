all: tt0 droplets

include ../makefile.this_machine

LIB_SRC = 	kernel3d_mpi.cpp knlmat3d_mpi.cpp dense3d_mpi.cpp let3d_mpi.cpp fmm3d_mpi.cpp matmgnt3d_mpi.cpp fmm3d_setup_mpi.cpp fmm3d_eval_mpi.cpp fmm3d_check_mpi.cpp manage_petsc_events.cpp mpi_workarounds.cpp

ifeq ($(USE_GPU),yes)
LIB_OBJ = 	$(LIB_SRC:.cpp=.o) gpu_setup.o
else
LIB_OBJ = 	$(LIB_SRC:.cpp=.o)
endif

DEP     = 	$(LIB_SRC:.cpp=.d)

libfmm3d_mpi.a:	$(LIB_OBJ)
	$(AR) $(ARFLAGS) libfmm3d_mpi.a $(LIB_OBJ)
	$(RANLIB) libfmm3d_mpi.a

tt0: libfmm3d_mpi.a tt0.o # chkopts
	${CLINKER} -o tt0 tt0.o libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 

droplets: libfmm3d_mpi.a droplets.o # chkopts
	${CLINKER} -o droplets droplets.o libfmm3d_mpi.a ../common/libcommon.a ${LIBS} ${PETSC_LIB} 
# -include $(DEP)

