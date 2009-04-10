#!/usr/local/bin/bash

WALLTIME=${WALLTIME-0:30:00}

OPTS=$1 ; shift
NP=$1 ; shift
PPN=$1 ; shift
OUTTAG=$1 ; shift

if \
  test -z "${OPTS}" \
  || test -z "${NP}" \
  || test -z "${PPN}" \
  || test -z "${OUTTAG}" \
; then
  echo \
"
usage: $0 <options-file> <num-nodes> <procs-per-node> <output-tag-name>

Set these environment variables to change defaults:
  WALLTIME=${WALLTIME}
  RUNSCRIPT=${RUNSCRIPT}
"
  exit 1
fi

RUNSCRIPT=${RUNSCRIPT-${OUTTAG}.pbs}

cat run.pbs.in \
  | sed "s,@WALLTIME@,${WALLTIME},g" \
  | sed "s,@NP@,${NP},g" \
  | sed "s,@PPN@,${PPN},g" \
  | sed "s,@OUTTAG@,${OUTTAG},g" \
  | sed "s,@OPTS@,${OPTS},g" \
  | sed "s,@LOGDIR@,${OUTTAG}--LOGS,g" \
  > "${RUNSCRIPT}"

echo \
"
Generated run script. To submit:

  qsub ${RUNSCRIPT}
"

# eof
