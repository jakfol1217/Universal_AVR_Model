# 1st argument - checkpoing_path,
# 2nd argument - slurm_id dependency (default=1)
# 3rd argument - additional hydra config (e.g. +increment_dataloader_idx=1)
test_prepare () {
    CHECKPOINT_PATH=$1
    LAST_ID=${2:-1}
    ADDITIONAL_PARAMS=${3}
    echo "Params:"
    echo $CHECKPOINT_PATH
    echo $LAST_ID
    echo $ADDITIONAL_PARAMS
}

# arugments - any number of tasks to run
test_run () {
    echo "Slurm ids:"
    for task_nm in ${@}; do
        LAST_ID=$(sbatch --parsable --time=0-00:30:00 --dependency=afterany:${L>
        echo -n "${LAST_ID} "
    done
    echo
}

test_bongard_logo () {
    test_prepare $1 $2 $3

    test_run bongard_logo_test_bd_vit_2 bongard_logo_test_ff_vit_2 bongard_logo>
}

test_vaec () {
    test_prepare $1 $2 $3

    test_run vaec_test1_vit_2 vaec_test2_vit_2 vaec_test3_vit_2 vaec_test4_vit_>
}

test_bongard_hoi () {
    test_prepare $1 $2 $3

    test_run bongard_hoi_seen-seen_vit_2 bongard_hoi_seen-unseen_vit_2 bongard_>
}



test_vasr () {
    test_prepare $1 $2 $3
    
    test_run vasr_vit_2
}