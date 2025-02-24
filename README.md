# ri4et
Repeated Interactions for Electron Transfer (ri4et)

## Requirements and Installation

ri4et.py is built on QuTiP, and has only been tested with QuTiP version 5.0.2.

We recommend creating a new conda environment for QuTiP following the QuTiP documentation: https://qutip.readthedocs.io/en/latest/installation.html

You can see the exact conda environment we used in `environment.yml`

## Usage

You can use `python ri4et.py --help` to see usage options.

As an example, this command starts a repeated interactions calculation of a donor-acceptor model system:
```
python ri4et.py \
    --job_type repeated_interactions \
    --coupling 0.1 \
    --lamda 1 \
    --kT 1 \
    --gamma 0.01 \
    --dE 3 \
    --num_ho_states 16 \
    --max_time 1000 \
    --dt_output 1 \
    --dt_ri 0.1 \
    > output.csv
```

See the `example_slurm_submissions` directory for examples of all possible job types using a SLURM scheduler. 
