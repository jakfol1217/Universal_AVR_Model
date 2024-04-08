Download data

```bash
srun -A mandziuk-lab --time 02:00:00 --cpus-per-task=4 --mem=4GB --pty bash -i
export $(cat .env | xargs)
./src/data/scripts/pgm.sh -r attr.rel.pairs -p /mnt/evafs/groups/mandziuk-lab/akaminski/datasets
```

Create hdf5py representation

```bash
sbatch scripts/create_hdf5.sh \
    /app/datasets_to_hdf5.py h5pyfy_pgm \
    /app/data/pgm/attr.rel.pairs /app/data/h5py/pgm/attr.rel.pairs ""
```
