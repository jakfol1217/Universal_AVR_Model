# Data generation and acquisition


In this module we download/generate multiple data sources.

## List of dataset

1. BongardHOI
2. BongardLOGO


## Instruction

To gather the data run the following commands

```bash
git submodule update --init --recursive
```

To download a single dataset run the following:

```bash
# Example bongard_hoi dataset download
# "./datasets" represents target folder where files will be stored
./scripts/bongard_hoi.sh ./datasets
```
