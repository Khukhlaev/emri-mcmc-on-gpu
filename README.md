# emri-mcmc-on-gpu

Simple script to perform mcmc computations with EMRI waveforms on gpu.

Sangria dataset path can be set in ```config.yaml```

## Usage

To start new run:


```
python main.py
```

To resume existing run change ```resume``` in ```config.yaml``` to ```True``` and ```save_files``` to path to save files, from which you want to resume

## Likelihood on cpu

Change constant XP to be equal to ```np``` at the beginning of ```likelihood.py```
 and change ```use_gpu``` in ```config.yaml``` to ```False```

## Copy all save files

All save files are stored in the same directory as ```main.py```. To move them to another directory put needed directory in ```copy_script``` and run

```
./copy_script
```
