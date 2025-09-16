# master-thesis-marlinde

Scripts and files for Marlinde's master thesis

## G1124251

Real event to analyze. Check that subdir for more information in the README over there.

## Software to use

A conda environment is available here: `/data/gravwav/twouters/miniconda3/envs/bilby_pipe_neural_prior`
This can be used to clone the environment (if that works, this is the easiest, if not, the manual install is needed). Info on cloning conda environments found [here](https://stackoverflow.com/questions/40700039/how-can-you-clone-a-conda-environment-into-the-base-root-environment). 

Some instructions for manual installation:

To analyze real events, we will use `bilby`, since then we can use the latest waveforms developed for low-mass mergers (BNS etc). However, we will use a custom bilby version in case the features we implemented there will be used at some point in the thesis. 
- `bilby`: https://github.com/ThibeauWouters/bilby/tree/neural_prior_bilby_pipe
- `bilby_pipe`: run `pip install bilby_pipe` for the latest version
- `astropy`: run `pip install astropy` for the latest version
- `numpy`: Seems there are issues with Numpy, so check version after installing. If above `2.0.0`, then run
```bash
pip uninstall numpy
```
and confirm with `y`. Then run
```bash
pip install 'numpy<2.0.0'
```

## Running bilby_pipe on the Nikhef cluster

A bit of magic is needed for this. Follow these steps (note: things like `outdir`, `G1124251` might change if you change some settings in the ini file -- this is just a template!):

1. Create a directory and put your desired config and prior files there. The config file should have these lines
```bash
accounting=ligo.prod.o4.cbc.pe.bilby
accounting-user=None
```
since otherwise `bilby_pipe` will complain.

2. Run `bilby_pipe config.ini`. This will create a new subdir, by default called `outdir`, which itself has a subdir called `submit`.

3. In the `*.submit` files, but not the one starting with `dag_`, a few changes are needed: (i) remove the line about the accounting stuff again, since otherwise the Nikhef cluster complains, (ii) add the following few stoomboot specific settings just above the `queue` statement:
```bash
# Stoomboot-specific settings
+USeOS = "el9"
+JobCategory = "medium"
```
To easily automate this, you can also copy-paste the following in your `.bashrc` file (located in your home directory):
```bash
fix_submit_files () {
    local dir=${1:-.}  # default to current dir if no arg
    for f in "$dir"/*.submit; do
        [ -e "$f" ] || continue  # skip if no match
        [[ $(basename "$f") == dag_* ]] && continue  # skip DAG files
        echo "Fixing $f"
        sed -i '/^accounting_group/d' "$f"
        sed -i '/^queue/i \
# Stoomboot-specific settings\n+USeOS          = "el9"\n+JobCategory    = "medium"' "$f"
    done
}
```
After doing that, run `source .bashrc` to activate this new function. With this, you can run from the head directory where files are stored `fix_submit_files outdir/submit` and the described changes will be done for you.

4. Launch the job: run `condor_submit_dag outdir/submit/dag_G1124251.submit`.

5. Wait a bit and enjoy the posterior