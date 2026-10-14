#!/usr/bin/env bash
# Fetch the official AdvancedIF judge code (CC BY-NC 4.0, not vendored) and the AdvancedIF data at pinned revisions.
# usage: bash evaluation/advancedif/fetch.sh     -> evaluation/advancedif/AdvancedIF/ (code), data/if_oss_full_data.csv
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
code_commit=f9d30137c4139d4d9af260ae28108b5afae828c0
data_revision=e20cba9b94b59c027dfab00b29244e8bc42e4ab4
data_sha256=9a54916200ab4b7baee81268b9ea03f35d5fdb9f829da90d03ec7e8e9a40fec7
if [[ ! -d "$here/AdvancedIF/.git" ]]; then
  git clone https://github.com/facebookresearch/AdvancedIF "$here/AdvancedIF"
fi
git -C "$here/AdvancedIF" checkout -q "$code_commit"
mkdir -p "$here/data"
curl -fL --retry 3 -o "$here/data/if_oss_full_data.csv" \
  "https://huggingface.co/datasets/facebook/AdvancedIF/resolve/$data_revision/if_oss_full_data.csv"
echo "$data_sha256  $here/data/if_oss_full_data.csv" | sha256sum -c
