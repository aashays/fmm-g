#!/usr/local/bin/bash

NP="$1" ; shift
OUTDIR="$1" ; shift

if \
  test -z "${NP}" \
  || test -z "${OUTDIR}" \
; then
  echo "
usage: $0 <num-points> <output-directory>
"
  exit 1
fi

TT0=./tt0
OPTS_IN=options_tune.in
OPTS_OUT_BASE=${OUTDIR}/options_tune
PTSMAX_SEQ="32 64 128 256"
PTSMAX_GPU="${PTSMAX_SEQ} 512 1024 2048 4096"

if ! mkdir -p ${OUTDIR} ; then
  echo "*** Couldn't create output directory, '${OUTDIR}'. ***"
  exit 1
fi

for PTSMAX in ${PTSMAX_SEQ} ; do
  RUNTAG=seq--${PTSMAX}
  OPTS_OUT=${OPTS_OUT_BASE}--${RUNTAG}
  LOG_OUT=${OUTDIR}/TUNE_LOG--${RUNTAG}.log
  cat ${OPTS_IN} \
    | sed "s,@GPU_ULIST@,,g" \
    | sed "s,@GPU_S2M@,,g" \
    | sed "s,@NP@,${NP},g" \
    | sed "s,@PTSMAX@,${PTSMAX},g" \
    > ${OPTS_OUT}
  date > ${LOG_OUT}
  ${TT0} -options_file ${OPTS_OUT} | tee -a ${LOG_OUT}
  date >> ${LOG_OUT}
done

for PTSMAX in ${PTSMAX_GPU} ; do
  RUNTAG=ulist--${PTSMAX}
  OPTS_OUT=${OPTS_OUT_BASE}--${RUNTAG}
  LOG_OUT=${OUTDIR}/TUNE_LOG--${RUNTAG}.log
  cat ${OPTS_IN} \
    | sed "s,@GPU_ULIST@,-gpu_ulist,g" \
    | sed "s,@GPU_S2M@,,g" \
    | sed "s,@NP@,${NP},g" \
    | sed "s,@PTSMAX@,${PTSMAX},g" \
    > ${OPTS_OUT}
  date > ${LOG_OUT}
  ${TT0} -options_file ${OPTS_OUT} | tee -a ${LOG_OUT}
  date >> ${LOG_OUT}
done

for PTSMAX in ${PTSMAX_GPU} ; do
  RUNTAG=ulist_s2m--${PTSMAX}
  OPTS_OUT=${OPTS_OUT_BASE}--${RUNTAG}
  LOG_OUT=${OUTDIR}/TUNE_LOG--${RUNTAG}.log
  cat ${OPTS_IN} \
    | sed "s,@GPU_ULIST@,-gpu_ulist,g" \
    | sed "s,@GPU_S2M@,-gpu_s2m,g" \
    | sed "s,@NP@,${NP},g" \
    | sed "s,@PTSMAX@,${PTSMAX},g" \
    > ${OPTS_OUT}
  date > ${LOG_OUT}
  ${TT0} -options_file ${OPTS_OUT} | tee -a ${LOG_OUT}
  date >> ${LOG_OUT}
done

# eof
