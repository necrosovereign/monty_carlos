// Copyright 2024 Vladimir Kharchev

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A simple and straightforward way to carry out Monte-Carlo simulations.
//!
//! Monty-Carlos is intended to be an easy to use and understand library for running Monte-Carlo
//! simulations. A common problem, when working with statistical tests, is not understanding what
//! exactly do the test results signify. Using [`MonteCarlo::simulate_pvalue`] probabilities of
//! wide classes of test statistics can be calculated in a reasonable time.
//!
//! # Examples
//!
//! Let's assume that, in a physical experiment, a quantity 67.0 was measured. Additionally,
//! theoretical calculation predict that this quantity is 63.2 and the random error of the
//! instrument is estimated to be 4.9. A calculation shows that the measured quantity is
//! 0.78 standard deviations from
//!
//! This probability can be calculated in the following way:
//! ```
//! use rand_distr::Normal;
//! use rand::Rng;
//! use rand::distributions::Distribution;
//! use monty_carlos::sample::Sample;
//! use monty_carlos::MonteCarlo;
//!
//! // The struct used for generating and evaluating the dataset at every iteration of the
//! // simulation
//! struct MeasuredSample {
//!     // The distribution from which the dataset is generated.
//!     distr: Normal<f64>,
//!     // The place to store the dataset
//!     measurement: f64,
//! }
//!
//! impl MeasuredSample {
//!     // The parameters of the normal distribution
//!     // The theoretical prediction of the measurement
//!     const MEAN: f64 = 63.2;
//!     // The estimated random error of the instrument
//!     const STDEV: f64 = 4.9;
//!
//!     fn new() -> Self {
//!         Self {
//!             distr: Normal::new(Self::MEAN, Self::STDEV).unwrap(),
//!             // A placeholder that is never read by the simulation
//!             measurement: 0.0,
//!         }
//!     }
//! }
//!
//! impl Sample for MeasuredSample {
//!     // This describes how to generate the dataset
//!     fn generate(&mut self, rng: &mut impl Rng) {
//!         // One value is sampled from the distribution
//!         self.measurement = self.distr.sample(rng);
//!     }
//!
//!     // This describe how to calculate the statistic of interest from the dataset
//!     fn evaluate(&self) -> f64 {
//!         // the statistic of interest is how many standard deviations is the generated value removed
//!         // from the theoretical prediction
//!         // i.e. |(x - μ)/σ|
//!         f64::abs((self.measurement - Self::MEAN) / Self::STDEV)
//!     }
//! }
//!
//! // The test statistic for our dataset is 0.78
//! let test_statistic = 0.78;
//!
//! let sample = MeasuredSample::new();
//! let mut simulation = MonteCarlo::new(sample);
//! // For the sake of example a small number of iterations is chosen
//! simulation.iterations = 100;
//!
//! // this should give a value of about 0.56
//! let pvalue = simulation.simulate_pvalue(test_statistic);
//! println!("The probability is {pvalue}");
//! ```
//! From this simulation and the code it's easy to conclude that the probability of a measurement
//! being closer to the theoretical prediction is about 56%.
//!
//! The usual example for Monte-Carlo simulation is calculating the value of π by randomly
//! generating points in a unit square. The probability of random point in the 1x1 square to fall
//! inside a quarter unit circle centered at a corner is equal to the ratio of their areas, which
//! is equal to π/4.
//!
//! This can be accomplished by implementing [Sample](crate::sample::Sample) in the following way:
//!
//! ```
//! use rand::Rng;
//! use rand::distributions::Distribution;
//! use monty_carlos::sample::Sample;
//! use monty_carlos::MonteCarlo;
//!
//! // The generation uses the standard distribution, so there is no need to store it.
//! struct PointSample {
//!     // A point in a unit square.
//!     point: [f64; 2],
//! }
//!
//! impl PointSample {
//!
//!     fn new() -> Self {
//!         Self {
//!             point: [0.0; 2],
//!         }
//!     }
//! }
//!
//! impl Sample for PointSample {
//!     fn generate(&mut self, rng: &mut impl Rng) {
//!         // Generate two coordinates from the interval [0, 1)
//!         self.point[0] = rng.gen();
//!         self.point[1] = rng.gen();
//!     }
//!
//!     fn evaluate(&self) -> f64 {
//!         // The statistic is the distance from the origin.
//!         self.point.iter().map(|&x| x * x).sum()
//!     }
//! }
//! // The test statistic is the radius of the quarter circle.
//! let radius = 1.0;
//!
//! let sample = PointSample::new();
//! let mut simulation = MonteCarlo::new(sample);
//! simulation.iterations = 100;
//!
//! let area = simulation.simulate_pvalue(radius);
//! println!("π/4 = {area}");
//! ```
//! Running this simulation gives a value of about 0.75, which is reasonably close to π/4 for such
//! a small amount of iterations. With a million iteration, a value 0.7854 can be obtained, which
//! is a good approximation.
#![deny(clippy::pedantic)]
#![deny(missing_docs)]

pub mod sample;

pub mod monte_carlo;
pub use monte_carlo::MonteCarlo;
