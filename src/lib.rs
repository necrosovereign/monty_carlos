#![deny(clippy::pedantic)]

use std::cmp;

use rand::SeedableRng;

mod fitting;
pub mod sample;
use sample::Sample;

struct CmlRatio {
    left: usize,
    total: usize,
}

impl CmlRatio {
    fn new() -> CmlRatio {
        CmlRatio { left: 0, total: 0 }
    }

    fn update(&mut self, ord: cmp::Ordering) {
        if let cmp::Ordering::Less = ord {
            self.left += 1;
        }
        self.total += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    fn ratio(&self) -> f64 {
        self.left as f64 / self.total as f64
    }
}

pub struct MonteCarlo<S> {
    pub iterations: usize,
    sample: S,
    rng: rand::rngs::StdRng,
}

impl<S: Sample> MonteCarlo<S> {
    fn simulate_iteration(&mut self) -> f64 {
        self.sample.generate(&mut self.rng);
        self.sample.evaluate()
    }

    pub fn simulate_pvalue(mut self, test_statistic: f64) -> f64 {
        let mut ratio = CmlRatio::new();
        for _ in 0..self.iterations {
            let statistic = self.simulate_iteration();
            ratio.update(statistic.total_cmp(&test_statistic));
        }
        ratio.ratio()
    }

    pub fn simulate_distribution(mut self) -> Vec<f64> {
        if self.iterations == 0 {
            return vec![];
        }
        let mut full_distribution: Vec<f64> = (0..self.iterations)
            .map(|_| self.simulate_iteration())
            .collect();
        if self.iterations < Self::DISTRIBUTION_SPLITS {
            return full_distribution;
        }
        full_distribution.sort_by(f64::total_cmp);
        let max_value = full_distribution.last().copied().into_iter();
        let skip = self.iterations / Self::DISTRIBUTION_SPLITS;
        full_distribution
            .into_iter()
            .step_by(skip)
            .chain(max_value)
            .collect()
    }
}

impl<S> MonteCarlo<S> {
    pub const DEFAULT_ITERATIONS: usize = 1_000_000;
    const DISTRIBUTION_SPLITS: usize = 100;

    pub fn new(sample: S) -> Self {
        Self {
            iterations: Self::DEFAULT_ITERATIONS,
            sample,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }
}
