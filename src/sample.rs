use crate::fitting::FittingDistribution;

pub trait Sample {
    fn generate(&mut self, rng: &mut impl rand::Rng);

    fn evaluate(&self) -> f64;
}

#[allow(clippy::module_name_repetitions)]
pub struct KSSample<D> {
    distr: D,
    samples: Box<[f64]>,
}

impl<D> Sample for KSSample<D>
where
    D: rand::distributions::Distribution<f64>,
    D: statrs::distribution::ContinuousCDF<f64, f64>,
{
    fn generate(&mut self, rng: &mut impl rand::Rng) {
        self.samples.fill_with(|| self.distr.sample(rng));
        self.samples.sort_by(f64::total_cmp);
    }

    fn evaluate(&self) -> f64 {
        self.dmax(&self.distr)
    }
}

impl<D> KSSample<D>
where
    D: rand::distributions::Distribution<f64>,
    D: statrs::distribution::ContinuousCDF<f64, f64>,
{
    fn dmax(&self, distr: &D) -> f64 {
        let n = self.samples.len();
        #[allow(clippy::cast_precision_loss)]
        let deviations = self.samples.windows(2).enumerate().map(|(i, xs)| {
            let left = f64::abs(distr.cdf(xs[0]) - (i + 1) as f64 / n as f64);
            let right = f64::abs(distr.cdf(xs[1]) - (i + 1) as f64 / n as f64);
            f64::max(left, right)
        });
        let middle_max = deviations.max_by(f64::total_cmp).unwrap();
        let left = distr.cdf(*self.samples.first().unwrap());
        let right = 1.0 - distr.cdf(*self.samples.last().unwrap());
        left.max(middle_max.max(right))
    }
}

impl<D> KSSample<D> {
    pub fn new(distr: D, num_samples: usize) -> Option<Self> {
        if num_samples == 0 {
            return None;
        }
        let samples = std::iter::repeat(0.0).take(num_samples).collect();
        Some(Self { distr, samples })
    }
}

#[allow(clippy::module_name_repetitions)]
pub struct LillieforsSample<D> {
    inner: KSSample<D>,
}

impl<D> LillieforsSample<D> {
    pub fn new(distr: D, num_samples: usize) -> Option<Self> {
        Some(Self {
            inner: KSSample::new(distr, num_samples)?,
        })
    }
}

impl<D> Sample for LillieforsSample<D>
where
    D: rand::distributions::Distribution<f64>,
    D: statrs::distribution::ContinuousCDF<f64, f64>,
    D: FittingDistribution,
{
    fn generate(&mut self, rng: &mut impl rand::Rng) {
        self.inner.generate(rng);
    }

    fn evaluate(&self) -> f64 {
        let sample_distr = D::fit(&self.inner.samples);
        self.inner.dmax(&sample_distr)
    }
}
