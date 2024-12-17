#!/usr/bin/env bash

run_config=jean-zay-a100-dev
model_config=base_audio_1k
data_root_dir=$SCRATCH/data
subsample_name=base

. parse_options.sh

other_params=""

if [ "$run_config" == "local" ]; then
	launcher_config=""
elif [ "$run_config" == "local_with_launcher" ]; then
	launcher_config="hydra/launcher=submitit_local +run_config=local"
	other_params="--multirun $other_params"
else
	launcher_config="hydra/launcher=submitit_slurm +run_config=$run_config"
	other_params="--multirun $other_params"
fi

python3 fairseq_cli/hydra_train.py \
	$other_params \
	--config-dir pantagruel/config \
	--config-name $model_config \
	task.data=$data_root_dir \
	+task.subsample_csv=$subsample_name \
	$launcher_config \
