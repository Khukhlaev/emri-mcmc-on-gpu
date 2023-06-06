# emri-mcmc-on-gpu

Simple script to perform mcmc computations with EMRI waveforms on gpu.

Sangria dataset should be in the same directory (or you should change sangria path in ```setup_with_sangria``` function in ```likelihood.py```)

In order to perform run without sangria just change ```setup_with_sangria``` to ```setup``` in ```main.py```

## Usage

To start new run:


```
python main.py --niter=20000 
```

To resume existing run (with save files in the same directory):

```
python main.py --niter=20000 --resume=1
```

See 

```
python main.py --help
```

for more options.

## Likelihood on cpu

Change constant XP to np at the beginning of ```likelihood.py```
 and then run

```
python main.py --niter=20000 --use_gpu=0
```

## Copy all save files

Change directory in ```copy_script``` and run

```
./copy_script
```
