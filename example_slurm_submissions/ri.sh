#!/bin/bash -l
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH -t 1-00:00:00

label=weakly_coupled
job_type=repeated_interactions
coupling=0.1
lamda=1
kT=1
gamma=0.01
num_ho_states=16
max_time=1000
dt_output=0.1
dt_ri=0.1

dE=3
dE=$(printf "%05.2f" "$dE")

output_name="${label}_results_${job_type}_dE${dE}_dt_ri${dt_ri}"

conda activate qutip-env

python -u ../ri4et.py \
    --job_type $job_type \
    --coupling $coupling \
    --lamda $lamda \
    --kT $kT \
    --gamma $gamma \
    --dE $dE \
    --num_ho_states $num_ho_states \
    --max_time $max_time \
    --dt_output $dt_output \
    --dt_ri $dt_ri \
    > ${output_name//./_}.csv
