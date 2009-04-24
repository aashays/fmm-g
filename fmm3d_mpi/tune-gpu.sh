#!/usr/local/bin/bash

OPTS_IN="$1" ; shift
RUN_ULIST="$1" ; shift
RUN_S2M="$1" ; shift
RUN_L2T="$1" ; shift
RUN_VLIST="$1" ; shift
N="$1" ; shift
NG="$1" ; shift
OUTDIR="$1" ; shift

if \
  test -z "${RUN_ULIST}" \
  || test -z "${RUN_S2M}" \
  || test -z "${RUN_L2T}" \
  || test -z "${RUN_VLIST}" \
  || test -z "${N}" \
  || test -z "${NG}" \
  || test -z "${OUTDIR}" \
; then
  echo "
usage: $0 <options.in> <ulist?> <s2m?> <l2t?> <vlist?> <num-points> <num-gpus> <output-directory>
"
  exit 1
fi

TT0_BIN=./tt0${EXEEXT}
TT0_RUN=
OPTS_OUT_BASE=${OUTDIR}/$(basename ${OPTS_IN})
PTSMAX_SEQ=${PTSMAX_SEQ-"32 64 128 256"}
PTSMAX_GPU=${PTSMAX_GPU-"${PTSMAX_SEQ} 512 1024 2048 4096"}

if ! test -d "${OUTDIR}" ; then
  if ! mkdir -p "${OUTDIR}" ; then
    echo "*** Couldn't create output directory, '${OUTDIR}'. ***"
    exit 1
  fi
fi

TAG=""
if test x"${RUN_ULIST}" = x"yes" ; then
  GPU_ULIST="-gpu_ulist"
  TAG="${TAG}ulist-"
else
  GPU_ULIST=""
fi

if test x"${RUN_S2M}" = x"yes" ; then
  GPU_S2M="-gpu_s2m"
  TAG="${TAG}s2m-"
else
  GPU_S2M=""
fi

if test x"${RUN_L2T}" = x"yes" ; then
  GPU_L2T="-gpu_l2t"
  TAG="${TAG}l2t-"
else
  GPU_L2T=""
fi

if test x"${RUN_VLIST}" = x"yes" ; then
  GPU_VLIST="-gpu_vlist"
  TAG="${TAG}vlist-"
else
  GPU_VLIST=""
fi

if test -n "${TAG}" ; then
  PTSMAX_LIST="${PTSMAX_GPU}"
else # empty tag
  TAG="cpu-"
  PTSMAX_LIST="${PTSMAX_SEQ}"
fi

if test ${NG} -gt 1 ; then
  NP=`wc -l ${PBS_NODEFILE} | cut -d'/' -f1`
  MV2_SRQ_SIZE=4000
  mvapich2-start-mpd
  TT0_RUN="mpirun -machinefile ${PBS_NODEFILE} -np ${NP}"
fi

for PTSMAX in ${PTSMAX_LIST} ; do
  RUNTAG=${TAG}-${PTSMAX}
  OPTS_OUT=${OPTS_OUT_BASE}--${RUNTAG}
  LOG_OUT=${OUTDIR}/TUNE_LOG--${RUNTAG}.log
  cat ${OPTS_IN} \
    | sed "s,@GPU_ULIST@,${GPU_ULIST},g" \
    | sed "s,@GPU_S2M@,${GPU_S2M},g" \
    | sed "s,@GPU_L2T@,${GPU_L2T},g" \
    | sed "s,@GPU_VLIST@,${GPU_VLIST},g" \
    | sed "s,@N@,${N},g" \
    | sed "s,@PTSMAX@,${PTSMAX},g" \
    > ${OPTS_OUT}
  date | tee ${LOG_OUT}
  ${TT0_RUN} ${TT0_BIN} -options_file ${OPTS_OUT} 2>&1 | tee -a ${LOG_OUT}
  date | tee -a ${LOG_OUT}
done

if test ${NG} -gt 1 ; then
  mpdallexit
fi

# eof
