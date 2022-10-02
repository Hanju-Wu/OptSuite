#!/bin/bash
SRC_DIR=$2
# valid param: double, single, complex16, complex8
SCALAR=$1
WD=${SRC_DIR}/${SCALAR}/Examples

# create soft link
ln -sf ${WD}/example.in example.in
ln -sf ${WD}/example_irl.in example_irl.in
ln -sf ${WD}/compare.in compare.in
ln -sf ${WD}/compare_irl.in compare_irl.in
ln -sf ${WD}/illc1850.rra illc1850.rra
ln -sf ${WD}/mhd1280b.cua mhd1280b.cua

# run example
echo "==Testing ${SCALAR} PRECISION version without implicit restart.=="
./example_${SCALAR} < example.in
echo "==Testing ${SCALAR} PRECISION version with implicit restart.=="
./example_irl_${SCALAR} < example_irl.in

# compare
#echo "==Comparing ${SCALAR} PRECISION results to reference.=="
#./compare_${SCALAR} < compare.in
#echo "==Comparing ${SCALAR} PRECISION IRL results to reference.=="
#./compare_${SCALAR} < compare_irl.in
