#!/usr/bin/env bash

# first arg as number of times to run the script (at least 2)
# second arg as sbatch args
# thrid arg as script to run
# rest of args as args to script
# example: ./scripts/run_bulk.sh 3 "--time=1-00:00:00 --mem=16GB --cpus-per-task=16 --gpus=1" scripts/run.sh src/main.py +experiment=bongard_hoi_vasr_images model=stsnv3 batch_size=4 lr=0.00005

num_runs=$1
sbatch_args=$2
script=$3
shift 3
args=$@

# create helper function to run sbatch
run_sbatch() {
    sbatch --parsable $sbatch_args $1 $script $args $2
}
echo "Num runs: $num_runs"
echo "sbatch_args: $sbatch_args"
echo "script: $script"
echo "args: $args"


LAST_JOB_ID=$(run_sbatch)
echo "First job id: $LAST_JOB_ID"
for ((i=2; i<=$num_runs; i++))
do
    LAST_JOB_ID=$(run_sbatch "--dependency=afterany:$LAST_JOB_ID" "checkpoint_path=/app/model_checkpoints/${LAST_JOB_ID}/last.ckpt")
    echo "Job id $i: $LAST_JOB_ID"
done
