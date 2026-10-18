#!/bin/zsh
# Species guidance 1.5 inside the cross-entropy loop (docs/species-guidance-loop.md): six cem_v3 runs, paired by seed.
#   nohup caffeinate -i -s zsh scripts/run_sp15loop.sh > /tmp/vgc-pilot/sp15loop.log 2>&1 &
cd /tmp/vgc-pilot/src; PY=/tmp/vgc-pilot/.venv/bin/python
export SUPERVISOR_LOG=/tmp/vgc-pilot/sp15loop_supervisor
for s in 4041 4042 4043; do
  for arm in plain species15; do
    args=(CEM_TAG=sp15loop_${arm}_s$s CEM_SEED=$s)
    [[ $arm == species15 ]] && args+=(CEM_G_SPECIES=1.5)
    echo "== sp15loop_${arm}_s$s $(date '+%m-%d %H:%M')"
    env "${args[@]}" $PY battle_supervisor.py cem_v3.py run --then final --then report
    cp ${SUPERVISOR_LOG}_run.log /tmp/vgc-pilot/sp15loop_${arm}_s${s}_run.log; : > ${SUPERVISOR_LOG}_run.log
  done
done
$PY cem_species15_loop.py report
echo SP15LOOP_DONE
