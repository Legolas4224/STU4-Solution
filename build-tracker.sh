#!/bin/bash

if [ ! -d "cpp-tracker "]; then
    git clone --recursive --branch barebones https://github.com/joosm4n/filtration-station.git cpp-tracker
fi

cd cpp-tracker
cmake -S . -B build
cmake --build build -j4
cp build/rock_tracker* ../.
