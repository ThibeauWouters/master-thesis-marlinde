# G1124251 (aka July 1st 2023 SSM trigger)

Information about the trigger:

- Trigger time: `1372281638.2548172`
- [Discussion Git issue page](https://git.ligo.org/cbc/action_items/-/issues/48)
- Also see [this issue](https://git.ligo.org/publications/o4/cbc/science_case_study_teams/-/issues/3)
- GraceDB playground entry is [here](https://gracedb-playground.ligo.org/events/G1124251/view/)
- Runs performed before by other LVK members, back when the trigger was first found:
    - Preliminary run [here](https://ldas-jobs.ligo.caltech.edu/~jacob.golomb/o4_pe/G1124251/run/outdir/pages/html/1688415003_G1124251_data0_1372281638-264_analysis_L1_merge_result_1688415003_G1124251_data0_1372281638-264_analysis_L1_merge_result_Config.html)
    - With precession and tides [here](https://ldas-jobs.ligo.caltech.edu/~jacob.golomb/o4_pe/G1124251/run_pv2nrtidalv2/outdir/pages/html/1688428570_G1124374_data0_1372281638-2548172_analysis_L1_merge_result_1688428570_G1124374_data0_1372281638-2548172_analysis_L1_merge_result.html)
    - Latest run is [this one](https://ldas-jobs.ligo.caltech.edu/~jacob.golomb/o4_pe/G1124251/run_fhigh2000/jul17/web/home.html)
- [Slides on the EOS PE](https://dcc.ligo.org/G2301311) presented back then
- Data downloaded from GWOSC: [here](https://gwosc.org/archive/links/O4a_16KHZ_R1/L1/1372281126/1372281643/simple/)

## Directory structure

There are a few directories:

- `pe`: Subdirectories here correspond to different PE runs, using `bilby_pipe`. 
- `postprocessing`: Some plotting scripts to generate plots from the PE runs. Currently, these make cornerplots of the posteriors and also create a 2D plot showing source-frame chirp mass vs the mass-weighted Lambda tilde parameter.
- `psd`: Scripts to compute the PSD of the event. Currently with PyCBC, but we might want to check BayesWave soon.