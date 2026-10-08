#!/bin/bash
cd /tmp/vgc-pilot/vgc-bench
export PYTHONPATH=/tmp/vgc-pilot/vgc-bench:/tmp/vgc-pilot/src
PY=/tmp/vgc-pilot/.venv/bin/python
PORTS=(8123 8124 8125 8126 8127 8128 8129)
K=$1; N=$2; C=$3
rm -f /tmp/vgc-pilot/sc_*.log
T0=$($PY -c "import time;print(time.time())")
for i in $(seq 0 $((K-1))); do
  $PY /tmp/vgc-pilot/src/scale_test.py ${PORTS[$i]} $N $C > /tmp/vgc-pilot/sc_$i.log 2>&1 &
done
wait
T1=$($PY -c "import time;print(time.time())")
TOT=$(grep -h '^PORT' /tmp/vgc-pilot/sc_*.log | awk '{s+=$4} END{print s+0}')
$PY -c "
k,n,c,tot,t0,t1=$K,$N,$C,$TOT,$T0,$T1
w=t1-t0
print(f'  workers={k:2d}  concurrency={c:3d}  battles={tot:5d}  wall={w:6.1f}s  ->  {tot/w:6.1f} battles/sec')"
