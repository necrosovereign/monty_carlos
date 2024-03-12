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
