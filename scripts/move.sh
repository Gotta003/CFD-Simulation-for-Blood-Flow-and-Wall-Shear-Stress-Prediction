#!/bin/bash
rsync -avP \
    --ignore-existing \
    --include="*/" \
    --include="*.vtp" \
    --include="*.vtu" \
    --exclude="*" \
    /data/simulation_db/ matteo.gottardelli@baldo.disi.unitn.it:/mnt/meditech/group3/simulations/

rsync -avP \
    --ignore-existing \
    --include="*/" \
    --include="*.vtp" \
    --include="*.vtu" \
    --exclude="*" \
    /home/group4/Challenge3/simulation_db/ matteo.gottardelli@baldo.disi.unitn.it:/mnt/meditech/group3/simulations/

rsync -avP \
    --exclude=".evar_env/" \
    --include="*/" \
    --include="*.sh" \
    --include="*.py" \
    --include="*.txt" \
    --include="*.job" \
    --exclude="*" \
    /home/group4/Challenge3/vtp_analysis/ matteo.gottardelli@baldo.disi.unitn.it:/mnt/meditech/group3/vtp_analysis/

rsync -avP \
    --ignore-existing \
    --include="*.npz" \
    --exclude="*" \
    matteo.gottardelli@baldo.disi.unitn.it:/mnt/meditech/group3/vtp_analysis/outputs/pointclouds/ /home/group4/Challenge3/vtp_analysis/outputs/pointclouds/

rsync -avP  matteo.gottardelli@baldo.disi.unitn.it:/mnt/meditech/group3/vtp_analysis/outputs/npz_checks/ /home/group4/Challenge3/vtp_analysis/outputs/npz_checks/