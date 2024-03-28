# Data generation and acquisition

In this module we download/generate multiple data sources.

## List of dataset

1. BongardHOI (48GB)
2. BongardLOGO - download requires `unzip` and `gdown` utilities. (2.4GB)
3. ~~CLEVR - download requires globus-cli (alternative tensorflow). However, we couldn't find a way to automate download of all files there - only single file download is allowed. This doesn't include folders.~~
4. ~~deepiq-ooo (isn't it too small 5mb zip?)~~
5. dopt (938M)
6. vaec (25GB)
7. dSprites-OOO (19MB - fine tune only)
8. g-set (not automated yet)
9. i-Raven (57GB; 8.0GB single regime)
10. MNS (downloading script prepared, generation - not yet) (21GB; 13/4.1/4.1)
11. PGM (requires google project to be created and GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET environment variables to be set before running) - before downloading verify (code) which dataset you want to download.
12. SVRT (https://fleuret.org/cgi-bin/gitweb/gitweb.cgi?p=svrt.git;a=tree - weird git client, hard to clone/download snapshot in cli, best to pass tar.gz repo snapshot)
13. VAP/LABC (same as in PGM env variables are required) (46GB; 6.7GB, 12.7GB, 8.9GB, 8.9GB, 8.8GB - compressed)
14. VASR (TODO looks big both generation (requires first downloading images and generating new ones - storing both may not be an option))
15. Sandia
16. KiloGram
17. ARC

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
