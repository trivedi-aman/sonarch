//! Band-limited (windowed-sinc) polyphase resampling.
//!
//! The filter design follows torchaudio's `functional.resample` exactly, so
//! results can be checked against it and against sonarch's PyTorch path
//! (used for GPU tensors). The ratio is reduced by the GCD of the two rates;
//! each of the `new` output phases has its own filter of `taps` coefficients.

use std::f64::consts::PI;

use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub enum Window {
    Hann,
    Kaiser(f64),
}

/// Filter settings. Wider filters and lower rolloff give steeper anti-aliasing
/// at the cost of speed.
#[derive(Clone, Copy, Debug)]
pub struct Quality {
    pub lowpass_filter_width: u32,
    pub rolloff: f64,
    pub window: Window,
}

impl Quality {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            // torchaudio's default settings.
            "fast" => Some(Self {
                lowpass_filter_width: 6,
                rolloff: 0.99,
                window: Window::Hann,
            }),
            "high" => Some(Self {
                lowpass_filter_width: 16,
                rolloff: 0.945,
                window: Window::Kaiser(8.555_178_356_811_5),
            }),
            // torchaudio's documented "kaiser_best" settings (librosa's former default).
            "best" => Some(Self {
                lowpass_filter_width: 64,
                rolloff: 0.947_593_716_739_959_6,
                window: Window::Kaiser(14.769_656_459_379_492),
            }),
            _ => None,
        }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let half_sq = (x / 2.0) * (x / 2.0);
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..500 {
        term *= half_sq / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

pub struct Kernel {
    /// Input samples consumed per block (reduced input rate).
    orig: usize,
    /// Output samples produced per block (reduced output rate).
    new: usize,
    /// Zero padding before the signal.
    width: usize,
    /// For each phase, the offset of its first non-negligible coefficient
    /// within the full (torchaudio-sized, `2 * width + orig`) filter, and
    /// those coefficients. Most of a phase's
    /// `taps` are (numerically) zero, so skipping them is a large speed-up
    /// for ratios such as 44.1 kHz -> 16 kHz.
    phases: Vec<(usize, Vec<f32>)>,
}

impl Kernel {
    pub fn new(orig_rate: u32, new_rate: u32, quality: Quality) -> Self {
        let g = gcd(u64::from(orig_rate), u64::from(new_rate));
        let orig = (u64::from(orig_rate) / g) as usize;
        let new = (u64::from(new_rate) / g) as usize;

        let lpw = f64::from(quality.lowpass_filter_width);
        let base_freq = orig.min(new) as f64 * quality.rolloff;
        let width = (lpw * orig as f64 / base_freq).ceil() as usize;
        let taps = 2 * width + orig;
        let scale = base_freq / orig as f64;
        let kaiser_norm = match quality.window {
            Window::Kaiser(beta) => bessel_i0(beta),
            Window::Hann => 1.0,
        };

        let mut phases = Vec::with_capacity(new);
        let mut full = vec![0.0_f64; taps];
        for phase in 0..new {
            let shift = -(phase as f64) / new as f64;
            for (tap, coeff) in full.iter_mut().enumerate() {
                let idx = (tap as f64 - width as f64) / orig as f64;
                let t = ((shift + idx) * base_freq).clamp(-lpw, lpw);
                let window = match quality.window {
                    Window::Hann => (t * PI / lpw / 2.0).cos().powi(2),
                    Window::Kaiser(beta) => {
                        bessel_i0(beta * (1.0 - (t / lpw).powi(2)).max(0.0).sqrt()) / kaiser_norm
                    }
                };
                let x = t * PI;
                let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
                *coeff = sinc * window * scale;
            }
            // Coefficients outside the filter's support are ~1e-17 (sin of a
            // multiple of pi); dropping them changes results by < 1e-12.
            let peak = full.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
            let keep = |c: &f64| c.abs() > peak * 1e-12;
            let first = full.iter().position(keep).unwrap_or(0);
            let last = full.iter().rposition(keep).unwrap_or(0);
            let coeffs = full[first..=last].iter().map(|&c| c as f32).collect();
            phases.push((first, coeffs));
        }
        Self {
            orig,
            new,
            width,
            phases,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.orig == self.new
    }

    /// Number of output samples for `n` input samples: ceil(n * new / orig).
    pub fn output_len(&self, n: usize) -> usize {
        ((n as u128 * self.new as u128).div_ceil(self.orig as u128)) as usize
    }

    /// Resample one channel. Runs in parallel over output blocks.
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len();
        let out_len = self.output_len(n);
        if out_len == 0 {
            return Vec::new();
        }

        // Same padding as torchaudio: `width` zeros before, `width + orig` after.
        let mut padded = vec![0.0_f32; self.width + n + self.width + self.orig];
        padded[self.width..self.width + n].copy_from_slice(input);

        let mut output = vec![0.0_f32; out_len];
        // Group blocks so each task does a meaningful amount of work.
        let work_per_block: usize = self.phases.iter().map(|(_, c)| c.len()).sum();
        let blocks_per_task = (65_536 / work_per_block.max(1)).max(1);
        let chunk = self.new * blocks_per_task;
        output
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(task, out)| {
                let first_block = task * blocks_per_task;
                for (offset, sample) in out.iter_mut().enumerate() {
                    let block = first_block + offset / self.new;
                    let phase = offset % self.new;
                    let (first, filter) = &self.phases[phase];
                    let start = block * self.orig + first;
                    *sample = dot(&padded[start..start + filter.len()], filter);
                }
            });
        output
    }
}

/// Dot product written so LLVM vectorizes it (8 independent accumulators).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0_f32; 8];
    let (chunks_a, rem_a) = a.as_chunks::<8>();
    let (chunks_b, rem_b) = b.as_chunks::<8>();
    for (x, y) in chunks_a.iter().zip(chunks_b) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in rem_a.iter().zip(rem_b) {
        sum += x * y;
    }
    sum
}

/// Resample planar audio (`channels` planes of `frames` samples, back to back).
pub fn resample_planar(
    samples: &[f32],
    channels: usize,
    frames: usize,
    kernel: &Kernel,
) -> (Vec<f32>, usize) {
    if kernel.is_identity() {
        return (samples.to_vec(), frames);
    }
    let out_frames = kernel.output_len(frames);
    let mut out = Vec::with_capacity(channels * out_frames);
    for ch in 0..channels {
        out.extend(kernel.apply(&samples[ch * frames..(ch + 1) * frames]));
    }
    (out, out_frames)
}

/// Average all channels into one.
pub fn mixdown(samples: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let mut out = vec![0.0_f32; frames];
    for ch in 0..channels {
        for (o, s) in out.iter_mut().zip(&samples[ch * frames..(ch + 1) * frames]) {
            *o += *s;
        }
    }
    let inv = 1.0 / channels as f32;
    out.iter_mut().for_each(|o| *o *= inv);
    out
}

/// Second-order IIR section, normalized so a0 = 1 (same as scipy.signal.lfilter).
#[derive(Clone, Copy)]
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
}

impl Biquad {
    fn new(b: [f64; 3], a: [f64; 3]) -> Self {
        Self {
            b: [b[0] / a[0], b[1] / a[0], b[2] / a[0]],
            a: [a[1] / a[0], a[2] / a[0]],
        }
    }

    /// RBJ high-shelf as used by BS.1770 (pyloudnorm's design formulas).
    fn high_shelf(gain_db: f64, q: f64, fc: f64, rate: f64) -> Self {
        let a = 10f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * fc / rate;
        let alpha = w0.sin() / (2.0 * q);
        let (cos, sqrt_a) = (w0.cos(), a.sqrt());
        Self::new(
            [
                a * ((a + 1.0) + (a - 1.0) * cos + 2.0 * sqrt_a * alpha),
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos),
                a * ((a + 1.0) + (a - 1.0) * cos - 2.0 * sqrt_a * alpha),
            ],
            [
                (a + 1.0) - (a - 1.0) * cos + 2.0 * sqrt_a * alpha,
                2.0 * ((a - 1.0) - (a + 1.0) * cos),
                (a + 1.0) - (a - 1.0) * cos - 2.0 * sqrt_a * alpha,
            ],
        )
    }

    fn high_pass(q: f64, fc: f64, rate: f64) -> Self {
        let w0 = 2.0 * PI * fc / rate;
        let alpha = w0.sin() / (2.0 * q);
        let cos = w0.cos();
        Self::new(
            [(1.0 + cos) / 2.0, -(1.0 + cos), (1.0 + cos) / 2.0],
            [1.0 + alpha, -2.0 * cos, 1.0 - alpha],
        )
    }

    /// Filter one sample (transposed direct form II); `state` holds z1, z2.
    #[inline]
    fn step(&self, state: &mut [f64; 2], input: f64) -> f64 {
        let out = self.b[0] * input + state[0];
        state[0] = self.b[1] * input - self.a[0] * out + state[1];
        state[1] = self.b[2] * input - self.a[1] * out;
        out
    }
}

/// Channel weights from ITU-R BS.1770. Up to 5 channels follow
/// [L, R, C, Ls, Rs]; 6 channels are 5.1 [L, R, C, LFE, Ls, Rs] (LFE ignored).
fn channel_weights(channels: usize) -> Option<Vec<f64>> {
    match channels {
        1..=5 => Some([1.0, 1.0, 1.0, 1.41, 1.41][..channels].to_vec()),
        6 => Some(vec![1.0, 1.0, 1.0, 0.0, 1.41, 1.41]),
        _ => None,
    }
}

pub const LOUDNESS_BLOCK_SECONDS: f64 = 0.4;

/// Integrated loudness in LUFS (ITU-R BS.1770-4 gating), matching pyloudnorm.
///
/// Returns `Err` with a message for unsupported channel counts or audio shorter
/// than one 400 ms block, and `-inf` for silence.
pub fn integrated_loudness(
    samples: &[f32],
    channels: usize,
    frames: usize,
    rate: u32,
) -> Result<f64, String> {
    let weights = channel_weights(channels)
        .ok_or_else(|| format!("loudness supports 1-6 channels, got {channels}"))?;
    let rate_f = f64::from(rate);
    if (frames as f64) < LOUDNESS_BLOCK_SECONDS * rate_f {
        return Err(format!(
            "loudness needs at least {LOUDNESS_BLOCK_SECONDS} s of audio, got {:.3} s",
            frames as f64 / rate_f
        ));
    }

    let shelf = Biquad::high_shelf(4.0, std::f64::consts::FRAC_1_SQRT_2, 1500.0, rate_f);
    let high_pass = Biquad::high_pass(0.5, 38.0, rate_f);

    let block = LOUDNESS_BLOCK_SECONDS;
    let step = 0.25; // 75 % overlap
    let duration = frames as f64 / rate_f;
    let num_blocks = (((duration - block) / (block * step)).round_ties_even() as usize) + 1;
    let norm = 1.0 / (block * rate_f);

    // Block boundaries exactly as pyloudnorm computes them (truncation included).
    let lo = |j: usize| (block * (j as f64 * step) * rate_f) as usize;
    let hi = |j: usize| ((block * (j as f64 * step + 1.0) * rate_f) as usize).min(frames);
    let mut bounds: Vec<usize> = (0..num_blocks)
        .flat_map(|j| [lo(j).min(frames), hi(j)])
        .collect();
    bounds.sort_unstable();
    bounds.dedup();
    let position = |b: usize| bounds.binary_search(&b).unwrap_or(0);

    // One streaming pass per channel: K-weight, square, and record the running
    // sum at each boundary. Memory is O(blocks), not O(samples).
    let power: Vec<Vec<f64>> = (0..channels)
        .into_par_iter()
        .map(|ch| {
            let input = &samples[ch * frames..(ch + 1) * frames];
            let mut cumulative = vec![0.0_f64; bounds.len()];
            let (mut s1, mut s2) = ([0.0_f64; 2], [0.0_f64; 2]);
            let mut acc = 0.0_f64;
            let mut next = 0;
            for (idx, &sample) in input.iter().enumerate() {
                while next < bounds.len() && bounds[next] == idx {
                    cumulative[next] = acc;
                    next += 1;
                }
                let y = high_pass.step(&mut s2, shelf.step(&mut s1, f64::from(sample)));
                acc += y * y;
            }
            cumulative[next..].fill(acc);
            (0..num_blocks)
                .map(|j| {
                    let (l, h) = (lo(j).min(frames), hi(j));
                    norm * (cumulative[position(h)] - cumulative[position(l.min(h))])
                })
                .collect()
        })
        .collect();

    let block_loudness = |j: usize| -> f64 {
        let sum: f64 = (0..channels).map(|i| weights[i] * power[i][j]).sum();
        -0.691 + 10.0 * sum.log10()
    };
    let gated_loudness = |selected: &[usize]| -> f64 {
        if selected.is_empty() {
            return f64::NEG_INFINITY;
        }
        let sum: f64 = (0..channels)
            .map(|i| {
                let mean =
                    selected.iter().map(|&j| power[i][j]).sum::<f64>() / selected.len() as f64;
                weights[i] * mean
            })
            .sum();
        -0.691 + 10.0 * sum.log10()
    };

    let levels: Vec<f64> = (0..num_blocks).map(block_loudness).collect();
    let absolute: Vec<usize> = (0..num_blocks).filter(|&j| levels[j] >= -70.0).collect();
    let relative_gate = gated_loudness(&absolute) - 10.0;
    let selected: Vec<usize> = (0..num_blocks)
        .filter(|&j| levels[j] > relative_gate && levels[j] > -70.0)
        .collect();
    Ok(gated_loudness(&selected))
}
