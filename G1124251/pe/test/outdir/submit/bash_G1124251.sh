#!/usr/bin/env bash

# G1124251_data0_1372281638-2548172_generation
# PARENTS 
# CHILDREN G1124251_data0_1372281638-2548172_analysis_L1
if [[ "G1124251_data0_1372281638-2548172_generation" == *"$1"* ]]; then
    echo "Running: /data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior/bin/bilby_pipe_generation outdir/G1124251_config_complete.ini --label G1124251_data0_1372281638-2548172_generation --idx 0 --trigger-time 1372281638.2548172"
    /data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior/bin/bilby_pipe_generation outdir/G1124251_config_complete.ini --label G1124251_data0_1372281638-2548172_generation --idx 0 --trigger-time 1372281638.2548172
fi

# G1124251_data0_1372281638-2548172_analysis_L1
# PARENTS G1124251_data0_1372281638-2548172_generation
# CHILDREN G1124251_data0_1372281638-2548172_analysis_L1_final_result
if [[ "G1124251_data0_1372281638-2548172_analysis_L1" == *"$1"* ]]; then
    echo "Running: /data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior/bin/bilby_pipe_analysis outdir/G1124251_config_complete.ini --detectors L1 --label G1124251_data0_1372281638-2548172_analysis_L1 --data-dump-file outdir/data/G1124251_data0_1372281638-2548172_generation_data_dump.pickle --sampler dynesty"
    /data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior/bin/bilby_pipe_analysis outdir/G1124251_config_complete.ini --detectors L1 --label G1124251_data0_1372281638-2548172_analysis_L1 --data-dump-file outdir/data/G1124251_data0_1372281638-2548172_generation_data_dump.pickle --sampler dynesty
fi

# G1124251_data0_1372281638-2548172_analysis_L1_final_result
# PARENTS G1124251_data0_1372281638-2548172_analysis_L1
# CHILDREN 
if [[ "G1124251_data0_1372281638-2548172_analysis_L1_final_result" == *"$1"* ]]; then
    echo "Running: /data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior/bin/bilby_result --result outdir/result/G1124251_data0_1372281638-2548172_analysis_L1_result.hdf5 --outdir outdir/final_result --extension hdf5 --max-samples 20000 --lightweight --save"
    /data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior/bin/bilby_result --result outdir/result/G1124251_data0_1372281638-2548172_analysis_L1_result.hdf5 --outdir outdir/final_result --extension hdf5 --max-samples 20000 --lightweight --save
fi

