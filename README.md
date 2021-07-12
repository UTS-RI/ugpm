author: le.gentil.cedric@gmail.com (Cedric)

# Unified Gaussian Preintegrated Measurements (UGPMs)

## and the Linear Preintegrated Measurements (LPMs)

This repository provides the C++ implementation of the preintegration methods presented in our RSS'21 paper titled [_Continuous Integration over SO(3) for IMU Preintegration_](http://www.roboticsproceedings.org/rss17/p078.pdf) (with video [here](https://youtu.be/4erKqw6S5o0), and poster [there](https://raw.githubusercontent.com/UTS-CAS/ugpm/main/ugpm_poster.pdf)). If you are using that code for any purpose, please cite the corresponding work as explained at the end of this page.

In short preintegration is a way to combine IMU data into pseudo measurements called _preintegrated measurements_. This is especially useful in the context of optimisation-based state estimation.

This repository contains the implementation of two novel methods that are the _UGPMs_ and _LPMs_, and an implementation of our previous work [_Gaussian Process Preintegration for Inertial-Aided Navigation Systems_](https://ieeexplore.ieee.org/document/8979155) (_GPMs_).



### Dependencies

This repository depends on *cmake* and *g++* for the compilation.

Libraries needed (boost doesn't need all, but shouldn't harm to have it all :) ):
```
sudo apt-get install libeigen3-dev
sudo apt-get install libboost-all-dev
sudo apt-get install libceres-dev
```

The code uses OpenMP but should be part of your compiler.



### Compile 

You will need to compile the code in a _build_ directory.

```bash
cd <this-repo>
mkdir build
cd build
cmake ../src
make
```



### Run

This repository contains two executables:

- `app/ugpm_demo`
- `app/paper_metrics`

###### ugpm_demo

This allows you to run the different preintegration methods (UGPM, LPM, GPM) with different parameters over simulated IMU data. It then compute the error with respect to the ground truth. The parameters are:

- `-m, --method` : choice of the preintegration method "ugpm", "lpm", or "gpm" (with "ugpm" being the default value)
- `-l, --length` : length of the integration window (2 seconds by default)
- `-q, --quantum` : controls the length of the chunks in the _per-chunk_ mode of the preintegrated measurements. If the quantum is negative, the `per-chunk` mode is deactivated.
- `-n, -nb_inference` : number of inference to compute to test/show the small marginal computational cost of additional inferences in the same integration window (useful when dealing with high framerate sensor fusion, e.g., lidar-inertial)
- `-t, --train` : flag to activate the hyper-parameter training of the Gaussian Processes when applicable
- `-j, -jacobian` : displays the Jacobian matrices produced by the preintegration method for postintegration correction vs. the numerical differentiation (most used for debugging, there is a full performance analysis of the postintegration corrections in the paper)
- `-h, --help`: produces a succinct help message

Here is a typical output of the proposed program (ran from the build repository):

```
./app/ugpm_demo -l 1 -m ugpm

Preintegration demonstration with UGPM
Time elapsed: 109.725 ms
Preintegration errors over window of 1:
  Rotation [deg] = 0.00974787
  Velocity [m/s] = 0.00370596
  Position [m]   = 0.00291148

Covariance
 6.03032e-08  4.13616e-10 -3.47978e-10            0            0            0            0            0            0
 4.13616e-10  4.70761e-08  -1.3584e-09            0            0            0            0            0            0
-3.47978e-10  -1.3584e-09  5.14124e-08            0            0            0            0            0            0
           0            0            0  1.33392e-05 -1.02802e-06  2.03985e-06            0            0            0
           0            0            0 -1.02802e-06  1.51109e-05  1.59617e-06            0            0            0
           0            0            0  2.03985e-06  1.59617e-06   9.5349e-06            0            0            0
           0            0            0            0            0            0   8.7608e-07 -1.40623e-07   2.0043e-07
           0            0            0            0            0            0 -1.40623e-07  1.12362e-06   -2.819e-08
           0            0            0            0            0            0   2.0043e-07   -2.819e-08  8.60975e-07

```



###### paper_metrics

This executable prints out the experiments' results of the paper in a pseudo-latex format. It takes parameters for the number of Monte Carlo runs and to select which experiment to run. Please refer to `./app/paper_metrics -h` for more information.





### Ideas and stuff to have in mind if you want to use in your system:

###### Choice of preintegration method

If your main constraint is the computation time, I recommend the use of the __LPMs__.

For the rest the __UGPMs__ per-chunk would be my method of choice (excellent accuracy in contained computation time). In my tests I generally use a quantum of 0.2 sec.



###### Preintegration object

In short, the overall use of the `ImuPreintegration` object corresponds to the instantiation of the object with the constructor `celib::ImuPreintegration(data, start_t, t, preint_opt, prior)` with:

- `data`: an `ImuData` structure defined in `library/include/common/types.h` (basically just to vectors of accelerometer and gyroscope data).
- `start_t`: a `double` that represents the timestamp of the beginning of the integration window
- `t`: a `std::vector<std::vector<double> >` that contains the timestamps at which you want to infer the preintegrated measurements. Each of the vectors in `t` needs to be of increasing order (the vector of vector thing allows for passing multiple series of timestamp to ease the data management in the case of multi-modal systems).
- `preint_opt`: a `PreintOption` structure that specifies the parameters of the preintegration method. Its definition can be found in `library/include/imu_preintegration/preintegration.h` 
- `prior`: a `PreintPrior` structure that specifies the prior knowledge of the IMU biases. Its definition can be found in `library/include/imu_preintegration/preintegration.h`. __Warning__: I didn't really test that feature yet. Might be bugged. (Let me know if there is an issue there).

Then, you can retrieve the preintegrated measurements using `celib::ImuPreintegration::get(index_1, index_2)` with `index_1` and `index_2` corresponding to the indexes in the vector of vector `t[index_1][index_2]`. The output will be a `PreintMeas` structure as defined in `library/include/common/types.h`



###### Heads-up

Among the things to keep in mind: the timing performances shown in the paper are based on 100Hz IMU data. Using faster IMU will imply more data therefore slower computations. Additionally, to perform optimally, the UGPMs (and GPMs) need a bit of data overlap between integration windows, in this example, I arbitrarily chose 0.15 sec which is pretty big. That can probably be reduced while maintaining high accuracy.

I "implemented" some minimum parallelisation using OpenMP but I am sure more efficient implementation are possible (even GPU acceleration should be possible especially if many inferences per integration window are needed, e.g. per-lidar-point preintegrated measurements). I don't plan to make this code computationally more efficient on my own, but happy to discuss/help anyone who wants.

Things I would check if I have spare time: while providing OK results in quantitative experiments, the Jacobians for the postintegration time-shift correction of the UGPM and GPM rotation parts seems a bit off. Not sure why, to be investigated.



### Erratum in the RSS paper

In Table II-c), the first cells of the last two rows (_"Fast Pos er."_ and _"Slow Pos er."_ ) should be swapped (only the first cell, not the full row).



### Citing

The _UGPMs_ and _LPMs_ have both been introduced in [_Continuous Integration over SO(3) for IMU Preintegration_](http://www.roboticsproceedings.org/rss17/p078.pdf)

```bibtex
@inproceedings{LeGentil2021,
	title={{Continuous Integration over SO(3) for IMU Preintegration}},
	author={{Le Gentil}, Cedric and {Vidal-Calleja}, Teresa},
  	booktitle={Robotics: Science and Systems},
	year={2021}
}
```



The _GPMs_ have been presented in [_Gaussian Process Preintegration for Inertial-Aided Navigation Systems_](https://ieeexplore.ieee.org/document/8979155)

```bibtex
@article{LeGentil2020,
	title = {{Gaussian Process Preintegration for Inertial-Aided State Estimation}},
	author = {{Le Gentil}, Cedric and Vidal-calleja, Teresa and Huang, Shoudong},
	journal = {IEEE Robotics and Automation Letters},
	number = {2},
	pages = {2108--2114},
	volume = {5},
	year = {2020} 
}
```

