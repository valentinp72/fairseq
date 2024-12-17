#!/usr/bin/env bash

mamba activate fairseq 

export ROOT_DIR=$(dirname -- "$( readlink -f -- "$0"; )";)
export FAIRSEQ_DIR=$ROOT_DIR

export PATH=$ROOT_DIR/bin:$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=$FAIRSEQ_DIR:$FAIRSEQ_DIR/examples:$PYTHONPATH
