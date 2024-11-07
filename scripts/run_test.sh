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
        LAST_ID=$(sbatch --parsable --time=0-00:30:00 --dependency=afterany:${LAST_ID} scripts/run.sh src/test.py "checkpoint_path='${CHECKPOINT_PATH}'" data/tasks=[${task_nm}] ${ADDITIONAL_PARAMS})
        echo -n "${LAST_ID} "
    done
    echo
}

test_bongard_logo () {
    test_prepare $1 $2 $3

    test_run bongard_logo_test_bd bongard_logo_test_ff bongard_logo_test_hd_comb bongard_logo_test_hd_novel
}

test_vaec () {
    test_prepare $1 $2 $3

    test_run vaec_test1 vaec_test2 vaec_test3 vaec_test4 vaec_test5
}

test_bongard_hoi () {
    test_prepare $1 $2 $3

    test_run bongard_hoi_seen-seen_vit bongard_hoi_seen-unseen_vit bongard_hoi_unseen-seen_vit bongard_hoi_unseen-unseen_vit
}

test_vasr () {
    test_prepare $1 $2 $3

    test_run vasr_vit
}