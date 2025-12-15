This is code to reproduce the results of the article "An accuracy-runtime trade-off comparison of scalable Gaussian process approximations for spatial data" by Filippo Rambelli and Fabio Sigrist, Seminar for Statistics, ETH Zurich.

In our work, we carried out a comparison between eight Gaussian process approximations. Specifically, we included Vecchia's approximation, covariance tapering, modified predictive process/ FITC, full-scale tapering, fixed rank kriging, the multiresolution approximation, the SPDE approach, and periodic embedding. The first four approximations are implemented in Python in the GPboost package, while the remaining four in different R libraries. 

The comparison was carried out on four real-world datasets, available in the 'data' folder, and on six types of simulated datasets. Due to the large size, we did not upload the synthetic datasets in the repository, however, it is possible to generate them by running the script simulate_data.R located in the 'code' folder. One can find the scripts for running the various approximations in the subfolder named as the dataset(s).

In the 'saved_values_exactGP' folder we also stored some results from exact computations on datasets with moderate sample size, particularly the house price dataset and the simulated datasets with sample size N=10'000.
