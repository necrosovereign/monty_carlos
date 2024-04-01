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

//! Contains the [`MonteCarlo`] struct.
//!

use std::cmp;

use rand::SeedableRng;
use statrs::statistics::Statistics;

use crate::sample::Sample;

/// A struct to keep track of the fraction of the values less then the test value.
struct CmlRatio {
    /// Number of smaller values
    left: usize,
    /// Total number of values
    total: usize,
}

impl CmlRatio {
    /// Returns the initial `CmlRatio`
    fn new() -> CmlRatio {
        CmlRatio { left: 0, total: 0 }
    }

    /// Updates the ratio. `ord` describes whether the new value is less, equal or greater than the
    /// test value.
    fn update(&mut self, ord: cmp::Ordering) {
        // Only the values that are strictly less increase the ratio.
        if let cmp::Ordering::Less = ord {
            self.left += 1;
        }
        self.total += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    /// Gives the ratio so far.
    fn ratio(&self) -> f64 {
        self.left as f64 / self.total as f64
    }
}

/// The struct that contains the state of a Monte-Carlo simulation.
///
/// [`MonteCarlo<S>`] repeatedly calls [`Sample::generate`] and [`Sample::evaluate`] methods of `S` and
/// calculates a result from the statistics returned by [`Sample::evaluate`].
///
/// The method [`MonteCarlo::simulate`] runs the simulation and return an instance of
/// [`monte_carlo::Distribution`](crate::monte_carlo::Distribution), the methods of which return
/// the quantities that can be calculated from the simulations.
///
/// The other methods of [`MonteCarlo`] call the corresponding methods on the result of
/// [`MonteCarlo::simulate`]. The exception is the method [`MonteCarlo::simulate_pvalue`].
/// `self.simulate_pvalue(x)` returns the same value as `self.simulate.pvalue(x)` but uses much
/// less memory.
pub struct MonteCarlo<S> {
    /// Number of iterations of the Monte-Carlo simulation.
    pub iterations: usize,
    sample: S,
    /// The random generator used by `sample.generate`
    rng: rand::rngs::StdRng,
}

impl<S: Sample> MonteCarlo<S> {
    /// Returns the statistic calculated from one iteration of a Monte-Carlo simulation.
    fn simulate_iteration(&mut self) -> f64 {
        self.sample.generate(&mut self.rng);
        self.sample.evaluate()
    }

    /// Runs a Monte-Carlo simulations and returns the fraction of iterations where
    /// the statistic is less than `test_statistic`.
    pub fn simulate_pvalue(mut self, test_statistic: f64) -> f64 {
        let mut ratio = CmlRatio::new();
        for _ in 0..self.iterations {
            let statistic = self.simulate_iteration();
            ratio.update(statistic.total_cmp(&test_statistic));
        }
        ratio.ratio()
    }

    /// Runs a Monte-Carlo simulation and returns a value representing the distribution of
    /// generated statistics.
    ///
    /// # Memory usage
    /// Use `self.simulate_pvalue(x)` instead of `self.simulate.pvalue(x)` to reduce memory usage.
    pub fn simulate(mut self) -> Distribution {
        let full_distribution: Vec<f64> = (0..self.iterations)
            .map(|_| self.simulate_iteration())
            .collect();
        Distribution::new(full_distribution)
    }

    /// Runs a Monte-Carlo simulation and returns the vector of quantiles for each whole percent in
    /// order.
    ///
    /// Currently the quantiles are not exact.
    ///
    /// Returns `None` if `self.iterations` is not big enough.
    ///
    /// Corresponds to [`Distribution::quantiles`].
    pub fn simulate_distribution(self) -> Option<Vec<f64>> {
        self.simulate().quantiles()
    }

    /// Runs a Monte-Carlo simulation and returns the value of the statistic, such that
    /// a statistic is less the returned values with probability `alpha`.
    ///
    /// Essentially the inverse to [`Self::simulate_pvalue`].
    ///
    /// Corresponds to [`Distribution::quantile_of`].
    pub fn simulate_statistic(self, alpha: f64) -> f64 {
        self.simulate().quantile_of(alpha)
    }

    /// Runs a Monte-Carlo simulation and returns the mean and the standard deviation from the
    /// generated statistics.
    ///
    /// Corresponds to [`Distribution::mean`] and [`Distribution::stdev`].
    pub fn simulate_mean_stdev(self) -> (f64, f64) {
        // Collect statistics from each iteration into a vector.
        let full_distribution = self.simulate();
        (full_distribution.mean(), full_distribution.stdev())
    }
}

impl<S> MonteCarlo<S> {
    /// The default value of `self.iterations`
    pub const DEFAULT_ITERATIONS: usize = 1_000_000;

    /// Constructs an instance for a Monte-Carlo simulation based on the implementation [Sample] by
    /// `S`.
    pub fn new(sample: S) -> Self {
        Self {
            iterations: Self::DEFAULT_ITERATIONS,
            sample,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }
}

/// The distribution of statistics generated by a Monte-Carlo simulation
pub struct Distribution {
    distr: Vec<f64>, // distr should be sorted.
}

impl Distribution {
    const DISTRIBUTION_SPLITS: usize = 100;

    fn new(mut distr: Vec<f64>) -> Self {
        distr.sort_by(f64::total_cmp);
        Self { distr }
    }

    /// Length of `distr`.
    fn count(&self) -> usize {
        self.distr.len()
    }

    /// Returns the probability of a value from the distribution being less that `test_statistic`
    #[must_use]
    pub fn pvalue(&self, test_statistic: f64) -> f64 {
        let index = self.distr.partition_point(|x| *x <= test_statistic);
        #[allow(clippy::cast_precision_loss)]
        return index as f64 / self.count() as f64;
    }

    /// Returns the vector of quantiles for each whole percent in order.
    ///
    /// Currently the quantiles are not exact.
    ///
    /// Returns `None` if `self.iterations` is not big enough.
    #[must_use]
    pub fn quantiles(&self) -> Option<Vec<f64>> {
        if self.count() < Self::DISTRIBUTION_SPLITS {
            // Not enough iterations to calculate the percentiles
            return None;
        }
        // The 100% quantile is the maximal value
        let max_value: Option<f64> = self.distr.last().copied();
        // Approximate distance between quantiles
        let skip = self.count() / Self::DISTRIBUTION_SPLITS;
        self.distr
            .iter()
            .step_by(skip) // Iterator of 0%-99% quantiles
            .copied()
            .chain(max_value) // Add the 100% quantile
            .collect::<Vec<_>>()
            .into()
    }

    /// Returns `alpha` quantile of the distribution.
    #[must_use]
    pub fn quantile_of(&self, alpha: f64) -> f64 {
        // Calculate the index corresponding to `alpha`
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let index: usize = (self.count() as f64 * alpha) as usize;
        self.distr[index]
    }

    /// Returns the mean of the distribution.
    #[must_use]
    pub fn mean(&self) -> f64 {
        Statistics::mean(&self.distr)
    }
    /// Returns the standard deviation of the distribution.
    #[must_use]
    pub fn stdev(&self) -> f64 {
        Statistics::std_dev(&self.distr)
    }
}
