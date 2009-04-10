#!/usr/local/bin/bash

OPTS_IN="$1" ; shift
RUN_ULIST="$1" ; shift
RUN_S2M="$1" ; shift
RUN_L2T="$1" ; shift
NP="$1" ; shift
OUTDIR="$1" ; shift

if \
  test -z "${RUN_ULIST}" \
  || test -z "${RUN_S2M}" \
  || test -z "${RUN_L2T}" \
  || test -z "${NP}" \
  || test -z "${OUTDIR}" \
; then
  echo "
usage: $0 <options.in> <ulist?> <s2m?> <l2t?> <num-points> <output-directory>
"
  exit 1
fi

TT0=./tt0
OPTS_OUT_BASE=${OUTDIR}/$(basename ${OPTS_IN})
PTSMAX_SEQ=${PTSMAX_SEQ-"32 64 128 256"}
PTSMAX_GPU=${PTSMAX_GPU-"${PTSMAX_SEQ} 512 1024 2048 4096"}

if ! mkdir -p ${OUTDIR} ; then
  echo "*** Couldn't create output directory, '${OUTDIR}'. ***"
  exit 1
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

if test -n "${TAG}" ; then
  PTSMAX_LIST="${PTSMAX_GPU}"
else # empty tag
  TAG="cpu-"
  PTSMAX_LIST="${PTSMAX_SEQ}"
fi

for PTSMAX in ${PTSMAX_LIST} ; do
  RUNTAG=${TAG}-${PTSMAX}
  OPTS_OUT=${OPTS_OUT_BASE}--${RUNTAG}
  LOG_OUT=${OUTDIR}/TUNE_LOG--${RUNTAG}.log
  cat ${OPTS_IN} \
    | sed "s,@GPU_ULIST@,${GPU_ULIST},g" \
    | sed "s,@GPU_S2M@,${GPU_S2M},g" \
    | sed "s,@GPU_L2T@,${GPU_L2T},g" \
    | sed "s,@NP@,${NP},g" \
    | sed "s,@PTSMAX@,${PTSMAX},g" \
    > ${OPTS_OUT}
  date > ${LOG_OUT}
  ${TT0} -options_file ${OPTS_OUT} | tee -a ${LOG_OUT}
  date >> ${LOG_OUT}
done

# eof
