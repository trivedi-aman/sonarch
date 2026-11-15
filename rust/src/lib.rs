//! sonarch's native engine, exposed to Python as `sonarch._core`.
//!
//! Decoding runs entirely in Rust (Symphonia) with the GIL released, so
//! several files can be decoded truly in parallel.

mod dsp;

use std::fs::File;
use std::path::Path;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::dsp::{Kernel, Quality, mixdown, resample_planar};

create_exception!(
    _core,
    UnsupportedError,
    PyException,
    "The native decoder can't handle this input; use the fallback decoder."
);
create_exception!(
    _core,
    NativeDecodeError,
    PyException,
    "The native decoder failed."
);

/// Decoded audio: `channels` planes of `frames` samples each, stored back to back.
struct Decoded {
    samples: Vec<f32>,
    channels: usize,
    frames: usize,
    sample_rate: u32,
}

enum Failure {
    /// Format or codec not handled natively; the caller should fall back.
    Unsupported(String),
    /// The input looks supported but is broken.
    Failed(String),
}

impl Failure {
    fn into_pyerr(self) -> PyErr {
        match self {
            Failure::Unsupported(msg) => UnsupportedError::new_err(msg),
            Failure::Failed(msg) => NativeDecodeError::new_err(msg),
        }
    }
}

fn classify(err: SymphoniaError) -> Failure {
    match err {
        SymphoniaError::Unsupported(what) => Failure::Unsupported(format!("unsupported: {what}")),
        SymphoniaError::IoError(e) => Failure::Failed(format!("I/O error: {e}")),
        other => Failure::Failed(other.to_string()),
    }
}

fn decode_file(path: &str, max_seconds: Option<f64>) -> Result<Decoded, Failure> {
    let file = File::open(path).map_err(|e| Failure::Failed(format!("can't open: {e}")))?;
    let stream = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(classify)?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| Failure::Unsupported("no audio track".into()))?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| Failure::Unsupported("no audio codec parameters".into()))?
        .clone();

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(&params, &AudioDecoderOptions::default())
        .map_err(classify)?;

    // Same rounding as Python's round(): half to even.
    let frames_for =
        |rate: u32| max_seconds.map(|s| (s * f64::from(rate)).round_ties_even() as usize);
    let mut sample_rate = params.sample_rate.unwrap_or(0);
    let mut max_frames = frames_for(sample_rate).filter(|_| sample_rate > 0);
    let mut planes: Vec<Vec<f32>> = Vec::new();
    if let (Some(n), Some(channels)) = (track.num_frames, params.channels.as_ref()) {
        let reserve = max_frames.map_or(n as usize, |m| m.min(n as usize));
        planes = (0..channels.count())
            .map(|_| Vec::with_capacity(reserve))
            .collect();
    }

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(Failure::Unsupported(
                    "stream parameters change mid-file".into(),
                ));
            }
            Err(e) => return Err(classify(e)),
        };
        if packet.track_id != track_id {
            continue;
        }

        let buffer = decoder.decode(&packet).map_err(classify)?;
        let spec_rate = buffer.spec().rate();
        let frames = buffer.frames();
        let num_planes = buffer.num_planes();
        if frames == 0 {
            continue;
        }

        let first_audio = planes.first().is_none_or(Vec::is_empty);
        if first_audio {
            // The decoder's output is authoritative for layout and rate.
            if planes.len() != num_planes {
                planes = (0..num_planes).map(|_| Vec::new()).collect();
            }
            if spec_rate != sample_rate {
                sample_rate = spec_rate;
                max_frames = frames_for(sample_rate);
            }
        } else if planes.len() != num_planes || spec_rate != sample_rate {
            return Err(Failure::Unsupported(
                "channel count or sample rate changes mid-file".into(),
            ));
        }

        let mut tails: Vec<&mut [f32]> = planes
            .iter_mut()
            .map(|plane| {
                let start = plane.len();
                plane.resize(start + frames, 0.0);
                &mut plane[start..]
            })
            .collect();
        buffer.copy_to_slice_planar(&mut tails);

        if max_frames.is_some_and(|limit| planes[0].len() >= limit) {
            break;
        }
    }

    if sample_rate == 0 {
        return Err(Failure::Failed("unknown sample rate".into()));
    }
    let channels = planes
        .len()
        .max(params.channels.as_ref().map_or(1, |c| c.count()));
    let mut frames = planes.first().map_or(0, Vec::len);
    if let Some(limit) = max_frames {
        frames = frames.min(limit);
    }

    let mut samples = Vec::with_capacity(channels * frames);
    if planes.is_empty() {
        samples.resize(channels * frames, 0.0);
    } else {
        for plane in &planes {
            samples.extend_from_slice(&plane[..frames]);
        }
    }
    Ok(Decoded {
        samples,
        channels,
        frames,
        sample_rate,
    })
}

/// Optional per-file processing done inside the worker thread.
#[derive(Clone, Copy)]
struct Post {
    mono: bool,
    target_rate: Option<u32>,
    quality: Quality,
}

fn parse_quality(name: &str) -> PyResult<Quality> {
    Quality::from_name(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "quality must be 'fast', 'high' or 'best', got {name:?}"
        ))
    })
}

fn postprocess(mut d: Decoded, post: Post) -> Decoded {
    if post.mono && d.channels > 1 {
        d.samples = mixdown(&d.samples, d.channels, d.frames);
        d.channels = 1;
    }
    if let Some(rate) = post.target_rate.filter(|&r| r != d.sample_rate) {
        let kernel = Kernel::new(d.sample_rate, rate, post.quality);
        let (samples, frames) = resample_planar(&d.samples, d.channels, d.frames, &kernel);
        d.samples = samples;
        d.frames = frames;
        d.sample_rate = rate;
    }
    d
}

fn to_numpy<'py>(py: Python<'py>, decoded: Decoded) -> PyResult<(Bound<'py, PyArray2<f32>>, u32)> {
    let array = Array2::from_shape_vec((decoded.channels, decoded.frames), decoded.samples)
        .map_err(|e| NativeDecodeError::new_err(e.to_string()))?;
    Ok((array.into_pyarray(py), decoded.sample_rate))
}

/// Decode one file (at most `max_seconds` of it) into a float32 array of shape
/// (channels, frames).
///
/// Returns `(array, sample_rate)`. Raises `UnsupportedError` when the format
/// or codec isn't handled natively, `NativeDecodeError` when decoding fails.
#[pyfunction]
#[pyo3(signature = (path, max_seconds=None))]
fn decode<'py>(
    py: Python<'py>,
    path: String,
    max_seconds: Option<f64>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, u32)> {
    let result = py.detach(|| decode_file(&path, max_seconds));
    to_numpy(py, result.map_err(Failure::into_pyerr)?)
}

/// Decode many files in parallel on a thread pool.
///
/// Returns a list with one entry per path: `(array, sample_rate)` on
/// success, or `(None, error_kind, message)` where `error_kind` is
/// `"unsupported"` or `"failed"`, so the caller can fall back per file.
#[pyfunction]
#[pyo3(signature = (
    paths, max_seconds=None, num_threads=None, sample_rate=None, mono=false, quality="high"
))]
#[allow(clippy::too_many_arguments)]
fn decode_many<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    max_seconds: Option<f64>,
    num_threads: Option<usize>,
    sample_rate: Option<u32>,
    mono: bool,
    quality: &str,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if sample_rate == Some(0) {
        return Err(PyValueError::new_err("sample_rate must be positive"));
    }
    let post = Post {
        mono,
        target_rate: sample_rate,
        quality: parse_quality(quality)?,
    };
    let run = |paths: &[String]| -> Vec<Result<Decoded, Failure>> {
        paths
            .par_iter()
            .map(|p| decode_file(p, max_seconds).map(|d| postprocess(d, post)))
            .collect()
    };
    let results = py.detach(|| match num_threads {
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map(|pool| pool.install(|| run(&paths)))
            .map_err(|e| e.to_string()),
        None => Ok(run(&paths)),
    });
    let results = results.map_err(NativeDecodeError::new_err)?;

    let mut out = Vec::with_capacity(results.len());
    for result in results {
        let item = match result {
            Ok(decoded) => to_numpy(py, decoded)?.into_pyobject(py)?.into_any(),
            Err(Failure::Unsupported(msg)) => (py.None(), "unsupported", msg)
                .into_pyobject(py)?
                .into_any(),
            Err(Failure::Failed(msg)) => (py.None(), "failed", msg).into_pyobject(py)?.into_any(),
        };
        out.push(item);
    }
    Ok(out)
}

/// Resample a C-contiguous float32 array of shape (channels, frames).
///
/// Output length is ceil(frames * new_rate / orig_rate), matching torchaudio.
#[pyfunction]
#[pyo3(signature = (array, orig_rate, new_rate, quality="high"))]
fn resample<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    orig_rate: u32,
    new_rate: u32,
    quality: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if orig_rate == 0 || new_rate == 0 {
        return Err(PyValueError::new_err("sample rates must be positive"));
    }
    let quality = parse_quality(quality)?;
    let shape = array.shape();
    let (channels, frames) = (shape[0], shape[1]);
    let samples = array
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?;
    let (out, out_frames) = py.detach(|| {
        let kernel = Kernel::new(orig_rate, new_rate, quality);
        resample_planar(samples, channels, frames, &kernel)
    });
    let array = Array2::from_shape_vec((channels, out_frames), out)
        .map_err(|e| NativeDecodeError::new_err(e.to_string()))?;
    Ok(array.into_pyarray(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(decode_many, m)?)?;
    m.add("UnsupportedError", m.py().get_type::<UnsupportedError>())?;
    m.add("NativeDecodeError", m.py().get_type::<NativeDecodeError>())?;
    Ok(())
}
