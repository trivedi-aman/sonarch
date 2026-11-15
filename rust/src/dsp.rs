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
