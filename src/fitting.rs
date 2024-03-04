use statrs::{distribution::Normal, statistics::Statistics};

pub(crate) trait FittingDistribution {
    fn fit(samples: &[f64]) -> Self;
}

impl FittingDistribution for Normal {
    fn fit(samples: &[f64]) -> Self {
        let mean = Statistics::mean(samples);
        let stdev = Statistics::std_dev(samples);
        Normal::new(mean, stdev).unwrap()
    }
}
