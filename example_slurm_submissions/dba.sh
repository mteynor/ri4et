#!/bin/bash -l
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=200GB
#SBATCH -t 1-00:00:00

label=dba
system=DBA
job_type=lindblad
coupling=0.1
kT=1
gamma=0.01
energy_d=1.5
energy_b1=0.5
energy_b2=-0.5
energy_a=-3.5
position_d=-1.5
position_b1=-0.5
position_b2=0.5
position_a=1.5
num_ho_states=16
max_time=10
dt_output=1

output_name="${label}_results_${job_type}"

conda activate qutip-env

python -u ../ri4et.py \
    --system $system \
    --job_type $job_type \
    --coupling $coupling \
    --kT $kT \
    --gamma $gamma \
    --energies $energy_d $energy_b1 $energy_b2 $energy_a \
    --positions $position_d $position_b1 $position_b2 $position_a \
    --num_ho_states $num_ho_states \
    --max_time $max_time \
    --dt_output $dt_output \
    > ${output_name//./_}.csv
