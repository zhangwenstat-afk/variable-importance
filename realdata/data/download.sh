#!/usr/bin/env bash
# Download the public raw data into $1 (default: ./raw). LUCAS is not public; see README.
set -euo pipefail
D=${1:-raw}; mkdir -p "$D"/nhanes "$D"/simsl "$D"/sonreb
# NHANES 2011-2016 (cycles G, H, I)
declare -A Y=([G]=2011 [H]=2013 [I]=2015)
for s in G H I; do for f in DEMO BMX BPX GHB GLU INS TRIGLY HDL DIQ MCQ OGTT; do
  [ "$f$s" = INSG ] && continue          # 2011-12 insulin is in GLU_G
  curl -sfL "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/${Y[$s]}/DataFiles/${f}_$s.xpt" -o "$D/nhanes/${f}_$s.xpt"
done; done
# IWPC warfarin subset in the R package simsl 0.2.1 (CRAN)
curl -sfL https://cran.r-project.org/src/contrib/Archive/simsl/simsl_0.2.1.tar.gz -o "$D/simsl/simsl_0.2.1.tar.gz" \
  || curl -sfL https://cran.r-project.org/src/contrib/simsl_0.2.1.tar.gz -o "$D/simsl/simsl_0.2.1.tar.gz"
tar -xzf "$D/simsl/simsl_0.2.1.tar.gz" -C "$D/simsl" simsl/data/warfarin.RData
# SonReb database (Zenodo record 15392443, version 2)
curl -sfL "https://zenodo.org/api/records/15392443/files/Complete%20SonReb%20Database.csv/content" -o "$D/sonreb/sonreb.csv"
