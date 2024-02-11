# Data generation and acquisition


In this module we download/generate multiple data sources.

## List of dataset

1. BongardHOI
2. BongardLOGO - download requires `unzip` and `gdown` utilities.
3. ~~CLEVR - download requires globus-cli (alternative tensorflow). However, we couldn't find a way to automate download of all files there - only single file download is allowed. This doesn't include folders.~~
4. ~~deepiq-ooo (isn't it too small 5mb zip?)~~
5. dopt
6. dSprites-OOO
7. g-set (not automated yet)
8. i-Raven
9. MNS (downloading script prepared, generation - not yet)
10. PGM (requires google project to be created and GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET environment variables to be set before running) - before downloading verify (code) which dataset you want to download.

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
