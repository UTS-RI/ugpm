author: le.gentil.cedric@gmail.com (Cedric)

# Unified Gaussian Preintegrated Measurements (UGPMs)

## and the Linear Preintegrated Measurements (LPMs)

__THIS VERSION IS ASSOCIATED TO OUR IJRR'23 PAPER__, if you are after the RSS'21 implementation, please check the `rss` branch

This repository provides the C++ implementation of the preintegration methods presented in our IJRR'23 paper titled [_Continuous Latent State Preintegration for Inertial-Aided Systems_](https://doi.org/10.1177/02783649231199537). If you are using that code for any purpose, please cite the corresponding work as explained at the end of this page.

In short preintegration is a way to combine IMU data into pseudo measurements called _preintegrated measurements_. This is especially useful in the context of optimisation-based state estimation.

This repository contains the implementation of the _UGPMs_ and _LPMs_ (extended from our RSS'21 paper).



### Dependencies

This repository depends on *cmake* and *g++* for the compilation.

Libraries needed (boost is only needed for the example code, not in the preintegration code):
```
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libceres-dev
```


### Compile 

You will need to compile the code in a _build_ directory.

```bash
cd <this-repo>
mkdir build
cd build
cmake ..
make
```



### Run

This repository contains two executables:

- `./ugpm_demo`
- `./ugpm_tests`

###### ugpm_demo

This allows you to run the different preintegration methods (UGPM, LPM, GPM) with different parameters over simulated IMU data. It then compute the error with respect to the ground truth. The parameters are:

- `-m, --method` : choice of the preintegration method "ugpm", "lpm", or "gpm" (with "ugpm" being the default value)
- `-l, --length` : length of the integration window (1 seconds by default)
- `-q, --quantum` : controls the length of the chunks in the _per-chunk_ mode of the preintegrated measurements. If the quantum is negative, the `per-chunk` mode is deactivated (-1 by default).
- `-n, -nb_inference` : number of inference to compute to test/show the small marginal computational cost of additional inferences in the same integration window (useful when dealing with high framerate sensor fusion, e.g., lidar-inertial)
- `-j, -jacobian` : displays the Jacobian matrices produced by the preintegration method for postintegration correction vs. the numerical differentiation (most used for debugging, there is a full performance analysis of the postintegration corrections in the paper)
- `-c, -correlate` : flag to activate the correlation of the covariance matrix
- `-h, --help`: produces a succinct help message

Here is a typical output of the proposed program (ran from the build repository):

```
./ugpm_demo

Preintegration demonstration with UGPM
Time elapsed: 48.836 ms
Preintegration errors over window of 2 seconds:
  Rotation [deg] = 0.00965282
  Velocity [m/s] = 0.00557632
  Position [m]   = 0.00952968

Covariance
 7.71467e-08  5.65872e-10  5.75752e-10  8.79113e-08  1.65319e-09  3.63768e-07 -3.53171e-07  9.62554e-07  7.69773e-07
 5.65872e-10  7.71667e-08  5.14963e-10  1.81363e-07 -3.13346e-07  6.90536e-07 -3.91895e-07 -1.17165e-06  1.76373e-06
 5.75752e-10  5.14963e-10  7.69021e-08 -1.64092e-08 -7.89181e-07 -2.69892e-07  4.24886e-07 -1.51773e-06 -7.40387e-07
 8.79113e-08  1.81363e-07 -1.64092e-08  1.12349e-05  -2.4541e-07  2.57008e-06  6.34687e-05  2.22228e-06  7.88625e-06
 1.65319e-09 -3.13346e-07 -7.89181e-07  -2.4541e-07  2.25064e-05 -1.66091e-07  1.20559e-06  0.000100579 -2.22998e-06
 3.63768e-07  6.90536e-07 -2.69892e-07  2.57008e-06 -1.66091e-07  2.14592e-05 -6.27162e-06 -4.01028e-06  9.68205e-05
-3.53171e-07 -3.91895e-07  4.24886e-07  6.34687e-05  1.20559e-06 -6.27162e-06  0.000488208   2.8271e-05 -4.11921e-06
 9.62554e-07 -1.17165e-06 -1.51773e-06  2.22228e-06  0.000100579 -4.01028e-06   2.8271e-05  0.000657667 -3.34832e-05
 7.69773e-07  1.76373e-06 -7.40387e-07  7.88625e-06 -2.22998e-06  9.68205e-05 -4.11921e-06 -3.34832e-05  0.000599425

```

###### ugpm_tests

This executable runs a few simple test to check different features of this implementation (it intentionally triggers some warnings). Refer to the source code in `ugpm_random_tests.cpp` for more information. 



### Ideas and stuff to have in mind if you want to use in your system:

###### Choice of preintegration method

If your main constraint is the computation time, I recommend the use of the __LPMs__.
For slow/normal scenarios, __LPMs__ and __UGPMs__ perform relatively similarly.
In more challenging setups, the __UGPMs__ provide better accuracy, and the __UGPMs (per chunk)__ is still quite fast regardless of the interval window.



###### Preintegration object

In short, the overall use of the `ImuPreintegration` object corresponds to the instantiation of the object with the constructor `ugpm::ImuPreintegration(data, start_t, t, preint_opt, prior)` with:

- `data`: an `ImuData` structure defined in `preint/types.h` (basically just two vectors of accelerometer and gyroscope data).
- `start_t`: a `double` that represents the timestamp of the beginning of the integration window
- `t`: a `std::vector<std::vector<double> >` that contains the timestamps at which you want to infer the preintegrated measurements. Each of the vectors in `t` needs to be of increasing order (the vector of vector thing allows for passing multiple series of timestamp to ease the data management in the case of multi-modal systems).
- `preint_opt`: a `PreintOption` structure that specifies the parameters of the preintegration method. Its definition can be found in `preint/types.h` 
- `prior`: a `PreintPrior` structure that specifies the prior knowledge of the IMU biases. Its definition can be found in `preint/types.h`.

Then, you can retrieve the preintegrated measurements using `ugpm::ImuPreintegration::get(index_1, index_2)` with `index_1` and `index_2` corresponding to the indexes in the vector of vector `t[index_1][index_2]`. The output will be a `PreintMeas` structure as defined in `preint/types.h`

Note that this version provides overloads of the constructor and `get` methods to allow the user to provide only one timestamp or a vector of timestamps for `t`.
I also included a way to account for the uncertainty of the biases in the covariance via optional arguments of the method `get` (can be useful for later use in optimisation).


###### Heads-up

The computation time is impacted in different ways. We invite the user to read the paper for better understanding of the different parameters/options.
To perform optimally, the UGPMs need a bit of data overlap between integration windows (8*state period in the code, can probably be changed).

Not all the features implemented have been fully tested, bugs are still possible.



### Citing

The _UGPMs_ and _LPMs_ have both been introduced in [_Continuous Latent State Preintegration for Inertial-Aided Systems_](https://doi.org/10.1177/02783649231199537)

```bibtex
@article{LeGentil2023,
	title={{Continuous Latent State Preintegration for Inertial-Aided Systems}},
	author={{Le Gentil}, Cedric and {Vidal-Calleja}, Teresa},
	journal={The International Journal of Robotics Research},
	year={2023},
	doi={10.1177/02783649231199537},
	publisher={Sage Publications}
}
```
