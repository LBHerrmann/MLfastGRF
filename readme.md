
# Description


The following C++ code realizes a fast algorithm to sample non-stationary Gaussian random fields indexed by spatial domains.
It is based on the so called "stochastic partial differential equation" approach 
and uses a sinc quadrature combined with a BPX preconditioned Finite Element Method. The implementation enables parallelized execution with distributed memory and is tested with up to 4m degrees of freedom. The computational cost depends essentially linear on the number of degrees of freedom.



# Publication

Lukas Herrmann, Kristin Kirchner, Christoph Schwab<br />
Multilevel approximation of Gaussian random fields: Fast simulation<br />
Mathematical Models and Methods in Applied Sciences, 30(1), pp. 181--223, 2020. <br />
[DOI](https://doi.org/10.1142/S0218202520500050)



# Requirements


This code is based on version 2 of the "Boundary Element Template Library" (BETL2). 
BETL2 may be used for non-commercial purposes. Information regarding licence and usage conditions may be found on the BETL2 webpage

[https://www.sam.math.ethz.ch/betl/](https://www.sam.math.ethz.ch/betl/)

The requirements for BETL2 are detailed in its documentation and uses standard packages such as <br />

BLAS <br />
Boost<br />
Eigen3<br />


Additional requirements are <br />
MPICH <br />
Boost.MPI<br />




# Installation


The installation of BETL2 is required and assumed in the following. 
Copy the repository into the Betl2 source folder next to the folder Betl2/Library. 
This folder will be the "root" folder in the following.

Change into the root directory and create a new folder which can be used to build.
Run the command:

```bash
mkdir build
cd build
cmake ../
```

Compile the executables by 

```bash
make betl2_fastGRF_err_sqrtM
make betl2_fastGRF_L2L2_err_polygon
make betl2_fastGRF_cpu_time
```

# Reproduce the results from the paper

Several executables need to be run in order to reproduce the results for Figures 1-4.<br />
For Figure 1:
Change to root directory and execute the commands: 

```bash
./source/betl2_fastGRF_err_sqrtM --mesh square --num 7
./source/betl2_fastGRF_err_sqrtM --mesh square --num 4
```

For Figure 2:
Change to root directory and execute the commands:

```bash
./source/betl2_fastGRF_cpu_time --mesh square --rep 40 --num 8 --beta 0.75
```
For Figure 3: 
Change to root directory and execute the commands:
NCPU=48 
```bash
./source/betl2_fastGRF_L2L2_err_polygon --mesh square --ncpu $NCPU --num 9 --beta 0.75 --kappa1 10 --kappa2 10
./source/betl2_fastGRF_L2L2_err_polygon --mesh square --ncpu $NCPU --num 9 --beta 0.75 --kappa1 20 --kappa2 200
./source/betl2_fastGRF_L2L2_err_polygon --mesh square --ncpu $NCPU --num 9 --beta 0.75 --kappa1 20 --kappa2 2000
./source/betl2_fastGRF_L2L2_err_polygon --mesh square --ncpu $NCPU --num 9 --beta 0.75 --kappa1 2000 --kappa2 2000
```

For Figure 4: Change to root directory and run the commands:

```bash
./source/betl2_fastGRF_L2L2_err_polygon --mesh square --ncpu $NCPU --num 9 --beta 1.5 --kappa1 10 --kappa2 10
./source/betl2_fastGRF_L2L2_err_polygon --mesh polygon --ncpu $NCPU --num 9 --beta 1.5 --kappa1 10 --kappa2 10
```

To create the figures, we require python3 with matplotlib. 
Change to root directory and run the commands: 
```bash
python3 results/sqrt_err/plot_sqrt_err.py
python3 results/cpu_time/plot_cpu_time.py
python3 results/beta_0.75/plot_square.py
python3 results/beta_1.50/plot_square_polygon.py
```


# Acknowledgement

The included code "genericMLMC_v1.1.0" is written by Robert Gantner and a priliminary version of library "genericMLQMC", see the corresponding publication with [DOI](https://doi.org/10.1145/2929908.2929915). 
It is used here for convenience.
