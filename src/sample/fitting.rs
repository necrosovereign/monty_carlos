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

#[allow(clippy::module_name_repetitions)]
/// A trait for probability distributions that can be constructed from a dataset.
pub trait FittingDistribution {
    /// Constructs `Self` from `samples` dataset in a way that approximates its distribution.
    fn fit(samples: &[f64]) -> Self;
}

impl FittingDistribution for Normal {
    /// Returns a normal distribution with the mean equal to the mean of `samples` and the standard
    /// deviation equal to the unbiased standard deviation of `samples`.
    fn fit(samples: &[f64]) -> Self {
        let mean = Statistics::mean(samples);
        let stdev = Statistics::std_dev(samples);
        Normal::new(mean, stdev).unwrap()
    }
}
