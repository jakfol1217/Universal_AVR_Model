To run training use the following (hydra-launcher plugin seems not to work with enroot :/):

```bash
sbatch scripts/run.sh src/main.py +experiment=test_mns_train
```
