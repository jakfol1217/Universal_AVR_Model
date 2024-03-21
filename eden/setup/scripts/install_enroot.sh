#!/usr/bin/env bash

arch=$(dpkg --print-architecture)
curl -fSsL -O "https://github.com/NVIDIA/enroot/releases/download/v3.4.1/enroot_3.4.1-1_${arch}.deb"
sudo apt install -y ./*.deb
