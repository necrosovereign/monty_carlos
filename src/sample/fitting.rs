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

//! Contains [`FittingDistribution`] trait and its implementors for conducting a Lilliefors test.
//!
//! Conducting a Lilliefors test requires a method to construct a distribution of the given kind
//! from the dataset. [`FittingDistribution`] is a way to describe such a method.
use statrs::{distribution::Normal, statistics::Statistics};

/// A trait to construct probability distribution from a dataset
///
/// The implementor should contain the information about which kind of probability distribution should be
/// constructed.
pub trait DistributionFit {
    /// The Distribution that will be constructed.
    type Distr;
    /// Constructs a probability distribution of the kind described by `self` with parameters
    /// determined by `samples`.
    ///
    /// The constructed distribution should approximate the distribution of `samples`.
    fn fit(&self, samples: &[f64]) -> Self::Distr;
}

/// An implementor of [`DistributionFit`] that constructs a normal distribution.
struct NormalFit;

impl DistributionFit for NormalFit {
    type Distr = Normal;

    /// Returns a normal distribution with the mean equal to the mean of `samples` and the standard
    /// deviation equal to the unbiased standard deviation of `samples`.
    fn fit(&self, samples: &[f64]) -> Normal {
        let mean = Statistics::mean(samples);
        let stdev = Statistics::std_dev(samples);
        Normal::new(mean, stdev).unwrap()
    }
}
