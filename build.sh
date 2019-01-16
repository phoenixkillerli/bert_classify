#!/bin/bash

# copy web app files
cp vocab.txt tokenization.py common_tool.py app.py build/web/

# copy model files
mkdir build/model/case_type
cp -R saved_model/* build/model/case_type/

# build docker images
cd build
sudo docker-compose up --build