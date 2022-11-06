#!/bin/bash 

# This script is used to run the trainer script on the server


git clone https://github.com/shirsho-12/cs4243_threat_detection.git
cd cs4243_threat_detection && git checkout shirsho
mv cs4243_threat_detection/scripts scripts
mv cs4243_threat_detection/run.py run.py
mv cs4243_threat_detection/run_triplet.py run_triplet.py
mkdir models

mkdir data
# mv *.zip data/
unzip -q drive-download-20221025T072443Z-001.zip
unzip -q drive-download-20221025T072443Z-002.zip
unzip -q drive-download-20221025T072443Z-003.zip

# If unzip above does not work use these
# unzip -q cs4243_dataset/drive-download-20221025T072443Z-001.zip
# unzip -q cs4243_dataset/drive-download-20221025T072443Z-002.zip
# unzip -q cs4243_dataset/drive-download-20221025T072443Z-003.zip

mv carrying data/carrying
mv threat data/threat
mv normal data/normal
rm -r *.zip

du -sh data
find data -type f | wc -l

python run.py | tee resnet_18.txt

python run_triplet.py | tee triplet.txt
