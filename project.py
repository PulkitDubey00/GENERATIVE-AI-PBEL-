
# ============================================================
# ROTATING MACHINERY HEALTH & VIBRATION DIAGNOSTIC SYSTEM
# ============================================================

# A Vibration analysts, certified from ISO 18436 needs around 5 to 6 hours for this job
# But our WebApp can do this work in less than 5 minutes. 

import io
import re
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import fft
from scipy import signal

import streamlit as st

from google import genai
from google.genai import types

from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    PageBreak,
)


st.set_page_config(
    page_title="Rotating Machinery Vibration Diagnostic System",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "⚡ Rotating Machinery Health & Vibration "
    "Diagnostic System"
)

st.caption(
    "FFT Spectrum Analysis • Harmonic Analysis • "
    "Bearing Diagnostics • AI Engineering Assistant"
)


# ============================================================
# SESSION STATE
# ============================================================

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = ""

if "ai_report" not in st.session_state:
    st.session_state.ai_report = None

if "pdf_report" not in st.session_state:
    st.session_state.pdf_report = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=np.nan):
    """Safely convert a value to float."""

    try:
        return float(value)
    except Exception:
        return default


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

def load_uploaded_file(file_bytes, filename):
    """Load CSV or Excel file from bytes."""

    if filename.lower().endswith(".csv"):

        return pd.read_csv(
            io.BytesIO(file_bytes)
        )

    if filename.lower().endswith(
        (".xlsx", ".xls")
    ):

        return pd.read_excel(
            io.BytesIO(file_bytes)
        )

    raise ValueError(
        "Unsupported file type. "
        "Please upload CSV or XLSX."
    )


# ------------------------------------------------------------
# TIME COLUMN PROCESSING
# ------------------------------------------------------------

def convert_time_to_seconds(df, time_column):
    """
    Convert selected time column to seconds from the
    beginning of the recording.
    """

    series = df[time_column]

    # Timestamp / datetime
    if (
        pd.api.types.is_datetime64_any_dtype(series)
        or "timestamp" in time_column.lower()
        or "datetime" in time_column.lower()
    ):

        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

        if parsed.notna().sum() < 2:
            raise ValueError(
                f"Could not parse '{time_column}' "
                "as timestamps."
            )

        seconds = (
            parsed - parsed.iloc[0]
        ).dt.total_seconds()

        return seconds.to_numpy(dtype=float)

    # Numeric time
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().sum() < 2:
        raise ValueError(
            f"Could not interpret '{time_column}' "
            "as time."
        )

    values = numeric.to_numpy(dtype=float)

    # Convert to relative time
    values = values - values[0]

    return values


# ------------------------------------------------------------
# DATA PREPARATION
# ------------------------------------------------------------

def prepare_signal(
    df,
    time_column,
    accel_column,
):
    """
    Clean and prepare time-series acceleration data.
    """

    work = df[
        [time_column, accel_column]
    ].copy()

    work["__time"] = convert_time_to_seconds(
        work,
        time_column,
    )

    work["__accel"] = pd.to_numeric(
        work[accel_column],
        errors="coerce",
    )

    work = work.dropna(
        subset=[
            "__time",
            "__accel",
        ]
    )

    if len(work) < 32:
        raise ValueError(
            "Not enough valid samples for FFT analysis."
        )

    # Remove infinite values
    work = work.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    work = work.dropna(
        subset=[
            "__time",
            "__accel",
        ]
    )

    # Sort by time
    work = work.sort_values(
        "__time"
    )

    # Remove duplicate timestamps
    work = work.drop_duplicates(
        subset="__time",
        keep="first",
    )

    time_data = work[
        "__time"
    ].to_numpy(dtype=float)

    accel_data = work[
        "__accel"
    ].to_numpy(dtype=float)

    if len(time_data) < 32:
        raise ValueError(
            "Not enough samples after cleaning."
        )

    dt = np.diff(time_data)

    dt = dt[
        np.isfinite(dt) &
        (dt > 0)
    ]

    if len(dt) == 0:
        raise ValueError(
            "Time values are invalid or not increasing."
        )

    median_dt = float(
        np.median(dt)
    )

    fs = 1.0 / median_dt

    # Check sampling uniformity
    dt_cv = (
        np.std(dt) /
        np.mean(dt)
        if np.mean(dt) > 0
        else np.inf
    )

    # If sampling is significantly irregular,
    # resample onto a uniform grid.
    if dt_cv > 0.01:

        uniform_time = np.arange(
            time_data[0],
            time_data[-1],
            median_dt,
        )

        if len(uniform_time) >= 32:

            uniform_accel = np.interp(
                uniform_time,
                time_data,
                accel_data,
            )

            time_data = uniform_time
            accel_data = uniform_accel

            fs = 1.0 / median_dt

    return (
        time_data,
        accel_data,
        fs,
        dt_cv,
    )


# ------------------------------------------------------------
# FFT ANALYSIS
# ------------------------------------------------------------

def calculate_fft(
    accel_data,
    fs,
):
    """
    Calculate single-sided Hann-windowed FFT.
    """

    x = np.asarray(
        accel_data,
        dtype=float,
    )

    x = signal.detrend(
        x,
        type="constant",
    )

    n = len(x)

    window = signal.windows.hann(
        n,
        sym=False,
    )

    coherent_gain = np.mean(window)

    windowed = x * window

    spectrum = fft.rfft(
        windowed
    )

    freqs = fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    amplitudes = (
        np.abs(spectrum)
        / (n * coherent_gain)
    )

    # Single-sided correction
    if len(amplitudes) > 2:

        amplitudes[1:-1] *= 2.0

    return (
        freqs,
        amplitudes,
        windowed,
    )


# ------------------------------------------------------------
# TIME-DOMAIN METRICES
# ------------------------------------------------------------

def calculate_time_metrics(
    accel_data,
    fs,
):
    """
    Calculate time-domain acceleration metrics and
    estimated RMS velocity.
    """

    x = np.asarray(
        accel_data,
        dtype=float,
    )

    x_detrended = signal.detrend(
        x,
        type="constant",
    )

    rms = float(
        np.sqrt(
            np.mean(
                x_detrended ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(x_detrended)
        )
    )

    peak_to_peak = float(
        np.ptp(x_detrended)
    )

    crest_factor = (
        peak / rms
        if rms > 0
        else 0.0
    )

    # Statistical metrics
    mean = np.mean(
        x_detrended
    )

    std = np.std(
        x_detrended
    )

    if std > 0:

        skewness = float(
            np.mean(
                (
                    x_detrended - mean
                ) ** 3
            )
            / std ** 3
        )

        kurtosis = float(
            np.mean(
                (
                    x_detrended - mean
                ) ** 4
            )
            / std ** 4
        )

    else:

        skewness = 0.0
        kurtosis = 0.0

    # --------------------------------------------------------
    # Velocity RMS
    #
    # Acceleration is in g.
    # Convert to m/s².
    # Frequency-domain integration:
    #
    # V(f) = A(f) / (2*pi*f)
    #
    # PSD-based integration avoids large low-frequency drift.
    # --------------------------------------------------------

    acceleration_ms2 = (
        x_detrended * 9.80665
    )

    nperseg = min(
        4096,
        len(acceleration_ms2),
    )

    if nperseg >= 32:

        f_psd, pxx = signal.welch(
            acceleration_ms2,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
        )

        velocity_psd = np.zeros_like(
            pxx
        )

        valid = f_psd >= 1.0

        velocity_psd[valid] = (
            pxx[valid]
            / (
                2.0
                * np.pi
                * f_psd[valid]
            ) ** 2
        )

        if len(f_psd) > 1:

            velocity_variance = np.trapezoid(
                velocity_psd,
                f_psd,
            )

            velocity_rms_mm_s = (
                np.sqrt(
                    max(
                        velocity_variance,
                        0.0,
                    )
                )
                * 1000.0
            )

        else:

            velocity_rms_mm_s = np.nan

    else:

        velocity_rms_mm_s = np.nan

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "velocity_rms_mm_s": velocity_rms_mm_s,
    }


# ------------------------------------------------------------
# DOMINANT PEAK DETECTION
# ------------------------------------------------------------

def detect_spectral_peaks(
    freqs,
    amplitudes,
    max_frequency=None,
):
    """
    Detect significant FFT peaks and return them sorted
    by amplitude.
    """

    f = np.asarray(
        freqs,
        dtype=float,
    )

    a = np.asarray(
        amplitudes,
        dtype=float,
    )

    mask = f > 0

    if max_frequency is not None:

        mask &= (
            f <= max_frequency
        )

    f2 = f[mask]
    a2 = a[mask]

    if len(a2) < 3:

        return pd.DataFrame(
            columns=[
                "Frequency_Hz",
                "Amplitude_g",
            ]
        )

    max_amp = np.max(a2)

    prominence = max(
        max_amp * 0.01,
        np.finfo(float).eps,
    )

    distance = max(
        1,
        len(a2) // 1000,
    )

    indices, properties = signal.find_peaks(
        a2,
        prominence=prominence,
        distance=distance,
    )

    if len(indices) == 0:

        return pd.DataFrame(
            columns=[
                "Frequency_Hz",
                "Amplitude_g",
            ]
        )

    peak_df = pd.DataFrame(
        {
            "Frequency_Hz":
                f2[indices],
            "Amplitude_g":
                a2[indices],
        }
    )

    peak_df = peak_df.sort_values(
        "Amplitude_g",
        ascending=False,
    )

    return peak_df.reset_index(
        drop=True
    )


# ------------------------------------------------------------
# HARMONIC ANALYSIS
# ------------------------------------------------------------

def find_harmonic_peak(
    freqs,
    amplitudes,
    target_frequency,
    tolerance_fraction=0.03,
):
    """
    Find the strongest spectral component around a target
    harmonic frequency.
    """

    if target_frequency <= 0:
        return None

    tolerance = max(
        target_frequency
        * tolerance_fraction,
        freqs[1] - freqs[0]
        if len(freqs) > 1
        else target_frequency * 0.01,
    )

    mask = (
        np.abs(
            freqs - target_frequency
        )
        <= tolerance
    )

    mask &= freqs > 0

    if not np.any(mask):
        return None

    local_indices = np.where(
        mask
    )[0]

    best_index = local_indices[
        np.argmax(
            amplitudes[local_indices]
        )
    ]

    return {
        "expected_frequency":
            float(target_frequency),

        "detected_frequency":
            float(freqs[best_index]),

        "amplitude":
            float(amplitudes[best_index]),
    }


# ------------------------------------------------------------
# BEARING FREQUENCIES
# ------------------------------------------------------------

def calculate_bearing_frequencies(
    rpm,
    number_of_elements,
    ball_diameter,
    pitch_diameter,
    contact_angle_deg,
):
    """
    Calculate:
        FTF
        BPFO
        BPFI
        BSF

    using standard rolling-element bearing equations.
    """

    fr = rpm / 60.0

    n = float(
        number_of_elements
    )

    bd = float(
        ball_diameter
    )

    pd = float(
        pitch_diameter
    )

    theta = np.deg2rad(
        contact_angle_deg
    )

    if (
        n <= 0
        or bd <= 0
        or pd <= 0
        or pd <= bd
    ):
        raise ValueError(
            "Invalid bearing geometry."
        )

    ratio = bd / pd

    cos_theta = np.cos(
        theta
    )

    ftf = (
        0.5
        * (
            1
            - ratio * cos_theta
        )
        * fr
    )

    bpfo = (
        n / 2.0
        * (
            1
            - ratio * cos_theta
        )
        * fr
    )

    bpfi = (
        n / 2.0
        * (
            1
            + ratio * cos_theta
        )
        * fr
    )

    bsf = (
        pd
        / (2.0 * bd)
        * (
            1
            - (
                ratio * cos_theta
            ) ** 2
        )
        * fr
    )

    return {
        "FTF": ftf,
        "BPFO": bpfo,
        "BPFI": bpfi,
        "BSF": bsf,
    }


# ------------------------------------------------------------
# BEARING PEAK MATCHING
# ------------------------------------------------------------

def match_bearing_frequencies(
    peak_df,
    bearing_frequencies,
):
    """
    Compare detected FFT peaks with bearing characteristic
    frequencies.
    """

    results = {}

    if (
        peak_df is None
        or peak_df.empty
    ):
        return results

    frequencies = peak_df[
        "Frequency_Hz"
    ].to_numpy()

    amplitudes = peak_df[
        "Amplitude_g"
    ].to_numpy()

    for name, target in bearing_frequencies.items():

        tolerance = max(
            target * 0.03,
            0.5,
        )

        differences = np.abs(
            frequencies - target
        )

        index = np.argmin(
            differences
        )

        if (
            differences[index]
            <= tolerance
        ):

            results[name] = {
                "Frequency_Hz":
                    float(
                        frequencies[index]
                    ),

                "Amplitude_g":
                    float(
                        amplitudes[index]
                    ),

                "Target_Hz":
                    float(target),

                "Difference_Hz":
                    float(
                        differences[index]
                    ),
            }

    return results


# ------------------------------------------------------------
# ENVELOPE SPECTRUM
# ------------------------------------------------------------

def calculate_envelope_spectrum(
    accel_data,
    fs,
    lowcut,
    highcut,
):
    """
    Calculate an envelope spectrum using a Butterworth
    band-pass filter followed by Hilbert transform.
    """

    nyquist = fs / 2.0

    lowcut = float(
        lowcut
    )

    highcut = float(
        highcut
    )

    if (
        lowcut <= 0
        or highcut >= nyquist
        or lowcut >= highcut
    ):
        raise ValueError(
            "Invalid envelope band. "
            f"Use 0 < lowcut < highcut < {nyquist:.2f} Hz."
        )

    sos = signal.butter(
        4,
        [
            lowcut / nyquist,
            highcut / nyquist,
        ],
        btype="bandpass",
        output="sos",
    )

    filtered = signal.sosfiltfilt(
        sos,
        signal.detrend(
            accel_data,
            type="constant",
        ),
    )

    analytic = signal.hilbert(
        filtered
    )

    envelope = np.abs(
        analytic
    )

    envelope = signal.detrend(
        envelope,
        type="constant",
    )

    frequencies, amplitudes, _ = calculate_fft(
        envelope,
        fs,
    )

    return (
        frequencies,
        amplitudes,
        envelope,
    )


# ------------------------------------------------------------
# FORMAT MARKDOWN FOR PDF
# ------------------------------------------------------------

def markdown_to_pdf_paragraphs(
    text,
    body_style,
    heading_style,
    bullet_style,
):
    """
    Convert basic Markdown AI output into ReportLab
    paragraphs.
    """

    elements = []

    text = str(text)

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:

            elements.append(
                Spacer(1, 5)
            )

            continue

        # Remove Markdown headings
        if line.startswith("#"):

            clean = line.lstrip(
                "#"
            ).strip()

            clean = clean.replace(
                "&",
                "&amp;",
            )

            elements.append(
                Paragraph(
                    clean,
                    heading_style,
                )
            )

            continue

        # Bullet
        if line.startswith(
            ("-", "*", "•")
        ):

            clean = line[1:].strip()

            clean = clean.replace(
                "&",
                "&amp;",
            )

            clean = clean.replace(
                "<",
                "&lt;",
            )

            clean = clean.replace(
                ">",
                "&gt;",
            )

            clean = re.sub(
                r"\*\*(.*?)\*\*",
                r"<b>\1</b>",
                clean,
            )

            elements.append(
                Paragraph(
                    "• " + clean,
                    bullet_style,
                )
            )

            continue

        clean = line.replace(
            "&",
            "&amp;",
        )

        clean = clean.replace(
            "<",
            "&lt;",
        )

        clean = clean.replace(
            ">",
            "&gt;",
        )

        clean = re.sub(
            r"\*\*(.*?)\*\*",
            r"<b>\1</b>",
            clean,
        )

        elements.append(
            Paragraph(
                clean,
                body_style,
            )
        )

    return elements


# ------------------------------------------------------------
# PDF REPORT GENERATOR
# ------------------------------------------------------------

def generate_pdf_report(
    ai_report,
    filename,
    rpm,
    sampling_rate,
    duration,
    num_samples,
    rms_accel,
    peak_accel,
    peak_to_peak,
    crest_factor,
    kurtosis,
    skewness,
    velocity_rms,
    harmonic_results,
    peak_df,
    bearing_frequencies,
    bearing_matches,
    plot_figure,
    envelope_figure=None,
):
    """
    Generate a professional engineering PDF report.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="Rotating Machinery Vibration Diagnostic Report",
        author="Rotating Machinery Vibration Diagnostic System",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey,
        spaceAfter=18,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
    )

    small_style = ParagraphStyle(
        "Small",
        parent=body_style,
        fontSize=7.5,
        leading=10,
        textColor=colors.grey,
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "ROTATING MACHINERY VIBRATION<br/>"
            "DIAGNOSTIC REPORT",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "FFT Spectrum Analysis • "
            "Harmonic Analysis • "
            "AI-Assisted Condition Assessment",
            subtitle_style,
        )
    )

    report_time = datetime.now().strftime(
        "%d %B %Y, %H:%M:%S"
    )

    story.append(
        Paragraph(
            f"<b>Source File:</b> "
            f"{filename}<br/>"
            f"<b>Report Generated:</b> "
            f"{report_time}",
            body_style,
        )
    )

    # ========================================================
    # 1. EXECUTIVE SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "1. Executive Summary",
            section_style,
        )
    )

    story.extend(
        markdown_to_pdf_paragraphs(
            ai_report,
            body_style,
            section_style,
            bullet_style,
        )[:8]
    )

    story.append(
        Spacer(1, 5)
    )

    # ========================================================
    # 2. MACHINE INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "2. Machine Information & Measurement Setup",
            section_style,
        )
    )

    machine_data = [
        ["Parameter", "Value"],
        [
            "Source File",
            filename,
        ],
        [
            "Operating Speed",
            f"{rpm:.2f} RPM",
        ],
        [
            "Running Speed",
            f"{rpm / 60.0:.3f} Hz",
        ],
        [
            "Sampling Rate",
            f"{sampling_rate:.3f} Hz",
        ],
        [
            "Number of Samples",
            f"{num_samples:,}",
        ],
        [
            "Recording Duration",
            f"{duration:.4f} s",
        ],
    ]

    table = Table(
        machine_data,
        colWidths=[
            2.8 * inch,
            3.2 * inch,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#1F4E78"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#F2F2F2"
                        ),
                    ],
                ),
            ]
        )
    )

    story.append(table)

    # ========================================================
    # 3. DATA QUALITY
    # ========================================================

    story.append(
        Paragraph(
            "3. Data Quality & Processing",
            section_style,
        )
    )

    story.append(
        Paragraph(
            "The uploaded time-series signal was cleaned for "
            "invalid values and duplicate timestamps. The "
            "sampling frequency was calculated from the "
            "time axis. A constant/trend component was removed "
            "before FFT processing and a Hann window was applied "
            "to reduce spectral leakage.",
            body_style,
        )
    )

    # ========================================================
    # 4. TIME-DOMAIN METRICS
    # ========================================================

    story.append(
        Paragraph(
            "4. Time-Domain Vibration Metrics",
            section_style,
        )
    )

    metrics_data = [
        ["Metric", "Measured Value"],
        [
            "RMS Acceleration",
            f"{rms_accel:.6f} g",
        ],
        [
            "Peak Acceleration",
            f"{peak_accel:.6f} g",
        ],
        [
            "Peak-to-Peak Acceleration",
            f"{peak_to_peak:.6f} g",
        ],
        [
            "Crest Factor",
            f"{crest_factor:.4f}",
        ],
        [
            "Kurtosis",
            f"{kurtosis:.4f}",
        ],
        [
            "Skewness",
            f"{skewness:.4f}",
        ],
        [
            "Estimated Velocity RMS",
            (
                f"{velocity_rms:.6f} mm/s"
                if np.isfinite(
                    velocity_rms
                )
                else "N/A"
            ),
        ],
    ]

    metrics_table = Table(
        metrics_data,
        colWidths=[
            3.0 * inch,
            2.5 * inch,
        ],
    )

    metrics_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#548235"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8.5,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#F2F2F2"
                        ),
                    ],
                ),
            ]
        )
    )

    story.append(metrics_table)

    # ========================================================
    # 5. FFT PLOT
    # ========================================================

    story.append(
        Paragraph(
            "5. Time-Domain Waveform & FFT Spectrum",
            section_style,
        )
    )

    plot_buffer = io.BytesIO()

    plot_figure.savefig(
        plot_buffer,
        format="png",
        dpi=180,
        bbox_inches="tight",
    )

    plot_buffer.seek(0)

    plot_image = PDFImage(
        plot_buffer,
        width=6.9 * inch,
        height=4.4 * inch,
    )

    story.append(plot_image)

    # ========================================================
    # 6. HARMONIC ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "6. Running-Speed Harmonic Analysis",
            section_style,
        )
    )

    harmonic_data = [
        [
            "Component",
            "Expected Hz",
            "Detected Hz",
            "Amplitude g",
        ]
    ]

    for name in [
        "1X",
        "2X",
        "3X",
    ]:

        result = harmonic_results.get(
            name
        )

        if result is None:

            harmonic_data.append(
                [
                    name,
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )

        else:

            harmonic_data.append(
                [
                    name,
                    f"{result['expected_frequency']:.3f}",
                    f"{result['detected_frequency']:.3f}",
                    f"{result['amplitude']:.6f}",
                ]
            )

    harmonic_table = Table(
        harmonic_data,
        colWidths=[
            1.0 * inch,
            1.5 * inch,
            1.5 * inch,
            1.5 * inch,
        ],
    )

    harmonic_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#7030A0"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(
        harmonic_table
    )

    # ========================================================
    # 7. DOMINANT FFT PEAKS
    # ========================================================

    story.append(
        Paragraph(
            "7. Dominant FFT Peaks",
            section_style,
        )
    )

    peak_data = [
        [
            "Rank",
            "Frequency (Hz)",
            "Amplitude (g)",
        ]
    ]

    if (
        peak_df is not None
        and not peak_df.empty
    ):

        for index, row in peak_df.head(
            12
        ).iterrows():

            peak_data.append(
                [
                    str(index + 1),
                    f"{row['Frequency_Hz']:.3f}",
                    f"{row['Amplitude_g']:.6f}",
                ]
            )

    else:

        peak_data.append(
            [
                "-",
                "No significant peaks",
                "-",
            ]
        )

    peak_table = Table(
        peak_data,
        colWidths=[
            0.8 * inch,
            2.0 * inch,
            2.0 * inch,
        ],
    )

    peak_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#C55A11"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey,
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    story.append(
        peak_table
    )

    # ========================================================
    # 8. BEARING ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "8. Bearing Characteristic Frequency Analysis",
            section_style,
        )
    )

    if bearing_frequencies:

        bearing_data = [
            [
                "Frequency",
                "Calculated Hz",
                "Detected Match",
                "Amplitude g",
            ]
        ]

        for name, freq in bearing_frequencies.items():

            match = bearing_matches.get(
                name
            )

            if match:

                bearing_data.append(
                    [
                        name,
                        f"{freq:.3f}",
                        f"{match['Frequency_Hz']:.3f}",
                        f"{match['Amplitude_g']:.6f}",
                    ]
                )

            else:

                bearing_data.append(
                    [
                        name,
                        f"{freq:.3f}",
                        "No close peak",
                        "-",
                    ]
                )

        bearing_table = Table(
            bearing_data,
            colWidths=[
                1.1 * inch,
                1.5 * inch,
                1.6 * inch,
                1.4 * inch,
            ],
        )

        bearing_table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#806000"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5,
                    ),
                ]
            )
        )

        story.append(
            bearing_table
        )

    else:

        story.append(
            Paragraph(
                "Bearing geometry was not supplied, so BPFO, "
                "BPFI, BSF and FTF could not be calculated.",
                body_style,
            )
        )

    # ========================================================
    # 9. ENVELOPE SPECTRUM
    # ========================================================

    if envelope_figure is not None:

        story.append(
            Paragraph(
                "9. Envelope Spectrum",
                section_style,
            )
        )

        envelope_buffer = io.BytesIO()

        envelope_figure.savefig(
            envelope_buffer,
            format="png",
            dpi=180,
            bbox_inches="tight",
        )

        envelope_buffer.seek(0)

        envelope_image = PDFImage(
            envelope_buffer,
            width=6.9 * inch,
            height=3.7 * inch,
        )

        story.append(
            envelope_image
        )

    # ========================================================
    # 10. AI DIAGNOSTIC REPORT
    # ========================================================

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "10. AI-Assisted Engineering Diagnostic",
            section_style,
        )
    )

    story.extend(
        markdown_to_pdf_paragraphs(
            ai_report,
            body_style,
            section_style,
            bullet_style,
        )
    )

    # ========================================================
    # 11. RECOMMENDED ACTIONS
    # ========================================================

    story.append(
        Paragraph(
            "11. Engineering Follow-Up",
            section_style,
        )
    )

    story.append(
        Paragraph(
            "The AI assessment should be treated as a "
            "decision-support result. Recommended follow-up "
            "measurements may include phase analysis, axial "
            "and radial vibration measurements, machine "
            "operating-condition checks, bearing inspection, "
            "lubrication assessment, alignment verification, "
            "balancing verification and comparison against "
            "historical vibration trends.",
            body_style,
        )
    )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "<b>Engineering Disclaimer:</b> "
            "This report is an AI-assisted engineering "
            "decision-support document. A spectral peak does "
            "not by itself prove a mechanical fault. Final "
            "maintenance or safety decisions must be based on "
            "appropriate engineering judgement, machine "
            "history, operating conditions, phase information, "
            "inspection results and applicable standards.",
            small_style,
        )
    )

    # ========================================================
    # BUILD
    # ========================================================

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()


# ------------------------------------------------------------
# AI ANALYSIS CONTEXT
# ------------------------------------------------------------

def build_analysis_context(
    rpm,
    fs,
    duration,
    metrics,
    harmonic_results,
    peak_df,
    bearing_frequencies,
    bearing_matches,
):
    """
    Build numerical context used by both the formal AI report
    and the interactive AI assistant.
    """

    peak_text = "No significant peaks detected."

    if (
        peak_df is not None
        and not peak_df.empty
    ):

        lines = []

        for _, row in peak_df.head(
            15
        ).iterrows():

            lines.append(
                f"- "
                f"{row['Frequency_Hz']:.3f} Hz : "
                f"{row['Amplitude_g']:.6f} g"
            )

        peak_text = "\n".join(
            lines
        )

    harmonic_text = ""

    for name in [
        "1X",
        "2X",
        "3X",
    ]:

        result = harmonic_results.get(
            name
        )

        if result:

            harmonic_text += (
                f"- {name}: "
                f"expected {result['expected_frequency']:.3f} Hz, "
                f"detected {result['detected_frequency']:.3f} Hz, "
                f"amplitude "
                f"{result['amplitude']:.6f} g\n"
            )

        else:

            harmonic_text += (
                f"- {name}: no close peak detected\n"
            )

    if bearing_frequencies:

        bearing_text = "\n".join(
            [
                f"- {name}: "
                f"{freq:.3f} Hz"
                for name, freq
                in bearing_frequencies.items()
            ]
        )

    else:

        bearing_text = (
            "No bearing geometry was supplied."
        )

    if bearing_matches:

        bearing_match_text = "\n".join(
            [
                f"- {name}: "
                f"detected {result['Frequency_Hz']:.3f} Hz "
                f"with amplitude "
                f"{result['Amplitude_g']:.6f} g"
                for name, result
                in bearing_matches.items()
            ]
        )

    else:

        bearing_match_text = (
            "No close bearing-frequency matches detected."
        )

    velocity_text = metrics[
        "velocity_rms_mm_s"
    ]

    if np.isfinite(
        velocity_text
    ):

        velocity_text = (
            f"{velocity_text:.6f} mm/s"
        )

    else:

        velocity_text = "N/A"

    context = f"""
============================================================
ROTATING MACHINERY VIBRATION ANALYSIS
============================================================

MEASUREMENT
-----------
Operating speed: {rpm:.3f} RPM
Running frequency: {rpm / 60.0:.5f} Hz
Sampling frequency: {fs:.3f} Hz
Recording duration: {duration:.5f} seconds

TIME-DOMAIN METRICS
-------------------
RMS acceleration: {metrics['rms']:.6f} g
Peak acceleration: {metrics['peak']:.6f} g
Peak-to-peak acceleration: {metrics['peak_to_peak']:.6f} g
Crest factor: {metrics['crest_factor']:.5f}
Kurtosis: {metrics['kurtosis']:.5f}
Skewness: {metrics['skewness']:.5f}
Estimated velocity RMS: {velocity_text}

RUNNING-SPEED HARMONICS
-----------------------
{harmonic_text}

DOMINANT FFT PEAKS
------------------
{peak_text}

BEARING CHARACTERISTIC FREQUENCIES
-----------------------------------
{bearing_text}

BEARING PEAK MATCHES
--------------------
{bearing_match_text}

ENGINEERING INTERPRETATION RULES
--------------------------------

1. A strong 1X component may be consistent with rotor
   unbalance, but 1X alone does not prove unbalance.

2. A strong 2X component may be associated with shaft or
   coupling misalignment, but 2X alone does not prove
   misalignment.

3. Multiple harmonics may be associated with mechanical
   looseness, but harmonic content alone does not prove
   looseness.

4. Bearing characteristic frequencies should be interpreted
   only when bearing geometry and machine speed are known.

5. Envelope-spectrum evidence can be useful for investigating
   rolling-element bearing defects.

6. Phase information, sensor direction, machine design,
   loading, temperature, lubrication and historical trends
   may be required for reliable fault identification.

7. Do not invent measurements that are not present.

8. Always distinguish:
   - Observation
   - Interpretation
   - Confidence
   - Recommended next check
"""

    return context


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Machine & Sensor Setup"
)


# ------------------------------------------------------------
# GEMINI API KEY
# ------------------------------------------------------------

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value="",
    help="Your Gemini API key is used only for this session.",
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

model_name = st.sidebar.text_input(
    "Gemini Model",
    value="gemini-3.6-flash",
)


# ------------------------------------------------------------
# RPM
# ------------------------------------------------------------

manual_rpm = st.sidebar.number_input(
    "Manual Operating Speed (RPM)",
    min_value=1.0,
    value=1780.0,
    step=10.0,
)


# ------------------------------------------------------------
# BEARING ANALYSIS
# ------------------------------------------------------------

st.sidebar.subheader(
    "🔩 Optional Bearing Geometry"
)

enable_bearing = st.sidebar.checkbox(
    "Enable Bearing Frequency Analysis",
    value=False,
)

bearing_elements = 0
bearing_ball_diameter = 0.0
bearing_pitch_diameter = 0.0
bearing_contact_angle = 0.0

if enable_bearing:

    bearing_elements = st.sidebar.number_input(
        "Number of Rolling Elements",
        min_value=1,
        value=8,
        step=1,
    )

    bearing_ball_diameter = st.sidebar.number_input(
        "Ball/Roller Diameter",
        min_value=0.001,
        value=10.0,
        step=0.5,
    )

    bearing_pitch_diameter = st.sidebar.number_input(
        "Pitch Diameter",
        min_value=0.001,
        value=50.0,
        step=1.0,
    )

    bearing_contact_angle = st.sidebar.number_input(
        "Contact Angle (degrees)",
        min_value=0.0,
        max_value=89.0,
        value=0.0,
        step=1.0,
    )


# ------------------------------------------------------------
# ENVELOPE ANALYSIS
# ------------------------------------------------------------

st.sidebar.subheader(
    "📈 Envelope Analysis"
)

enable_envelope = st.sidebar.checkbox(
    "Enable Envelope Spectrum",
    value=False,
)

envelope_low = 500.0
envelope_high = 4500.0

if enable_envelope:

    envelope_low = st.sidebar.number_input(
        "Envelope Band Low (Hz)",
        min_value=1.0,
        value=500.0,
        step=100.0,
    )

    envelope_high = st.sidebar.number_input(
        "Envelope Band High (Hz)",
        min_value=10.0,
        value=4500.0,
        step=100.0,
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.header(
    "1️⃣ Upload Vibration Data"
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel Accelerometer Log",
    type=[
        "csv",
        "xlsx",
        "xls",
    ],
)


# ------------------------------------------------------------
# SAVE FILE TO SESSION STATE
# ------------------------------------------------------------

if uploaded_file is not None:

    st.session_state.uploaded_bytes = (
        uploaded_file.getvalue()
    )

    st.session_state.uploaded_filename = (
        uploaded_file.name
    )


if (
    st.session_state.uploaded_bytes is None
    or st.session_state.uploaded_filename is None
):

    st.info(
        "Upload a CSV or Excel vibration file to begin."
    )

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_uploaded_file(
        st.session_state.uploaded_bytes,
        st.session_state.uploaded_filename,
    )

except Exception as exc:

    st.error(
        f"Could not load the file: {exc}"
    )

    st.stop()


if df.empty:

    st.error(
        "The uploaded file contains no data."
    )

    st.stop()


# ============================================================
# DATA PREVIEW
# ============================================================

st.subheader(
    "Raw Data Preview"
)

st.dataframe(
    df.head(10),
    use_container_width=True,
)


# ============================================================
# COLUMN DETECTION
# ============================================================

all_columns = list(
    df.columns
)

numeric_columns = list(
    df.select_dtypes(
        include=np.number
    ).columns
)


# ------------------------------------------------------------
# TIME COLUMN
# ------------------------------------------------------------

time_candidates = [
    col
    for col in all_columns
    if (
        "time" in str(col).lower()
        or "timestamp"
        in str(col).lower()
        or "datetime"
        in str(col).lower()
    )
]

if not time_candidates:

    time_candidates = [
        col
        for col in numeric_columns
    ]


default_time_index = 0

if "Time_s" in all_columns:

    default_time_index = (
        all_columns.index(
            "Time_s"
        )
    )

elif time_candidates:

    default_time_index = (
        all_columns.index(
            time_candidates[0]
        )
    )

time_column = st.selectbox(
    "Time / Timestamp Column",
    all_columns,
    index=default_time_index,
)


# ------------------------------------------------------------
# ACCELERATION COLUMN
# ------------------------------------------------------------

accel_candidates = [
    col
    for col in numeric_columns
    if (
        "accel" in str(col).lower()
        or "vibration"
        in str(col).lower()
    )
]

if not accel_candidates:

    accel_candidates = numeric_columns


if not accel_candidates:

    st.error(
        "No numeric acceleration/vibration column "
        "was found."
    )

    st.stop()


default_accel_index = 0

if "Accel_g" in accel_candidates:

    default_accel_index = (
        accel_candidates.index(
            "Accel_g"
        )
    )

elif "Vibration_X_g" in accel_candidates:

    default_accel_index = (
        accel_candidates.index(
            "Vibration_X_g"
        )
    )


accel_column = st.selectbox(
    "Acceleration / Vibration Column",
    accel_candidates,
    index=default_accel_index,
)


# ============================================================
# RPM SOURCE
# ============================================================

rpm_columns = [
    col
    for col in numeric_columns
    if (
        "rpm" in str(col).lower()
        or "speed" in str(col).lower()
    )
]

rpm_source_options = [
    "Manual RPM"
]

if rpm_columns:

    rpm_source_options.append(
        "RPM Column"
    )

rpm_source = st.selectbox(
    "Operating Speed Source",
    rpm_source_options,
)


rpm_column = None

if rpm_source == "RPM Column":

    rpm_column = st.selectbox(
        "RPM Column",
        rpm_columns,
    )


# ============================================================
# PREPARE SIGNAL
# ============================================================

try:

    (
        time_data,
        accel_data,
        sampling_rate,
        dt_cv,
    ) = prepare_signal(
        df,
        time_column,
        accel_column,
    )

except Exception as exc:

    st.error(
        f"Data preparation failed: {exc}"
    )

    st.stop()


# ============================================================
# DETERMINE RPM
# ============================================================

if (
    rpm_source == "RPM Column"
    and rpm_column is not None
):

    rpm_series = pd.to_numeric(
        df[rpm_column],
        errors="coerce",
    )

    rpm_series = rpm_series.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(rpm_series) == 0:

        st.error(
            "RPM column does not contain valid numeric data."
        )

        st.stop()

    rpm = float(
        rpm_series.median()
    )

else:

    rpm = float(
        manual_rpm
    )


if rpm <= 0:

    st.error(
        "Operating RPM must be greater than zero."
    )

    st.stop()


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

duration = float(
    time_data[-1]
    - time_data[0]
)

number_of_samples = len(
    accel_data
)

one_x_hz = rpm / 60.0
two_x_hz = one_x_hz * 2.0
three_x_hz = one_x_hz * 3.0


# ============================================================
# DISPLAY OPERATING INFORMATION
# ============================================================

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "Operating Speed",
    f"{rpm:.1f} RPM",
)

info2.metric(
    "Sampling Rate",
    f"{sampling_rate:.1f} Hz",
)

info3.metric(
    "Samples",
    f"{number_of_samples:,}",
)

info4.metric(
    "Duration",
    f"{duration:.3f} s",
)


st.markdown(
    f"""
**Running-speed references**

- 1× = `{one_x_hz:.3f} Hz`
- 2× = `{two_x_hz:.3f} Hz`
- 3× = `{three_x_hz:.3f} Hz`
"""
)


# ============================================================
# DATA QUALITY WARNING
# ============================================================

if dt_cv > 0.01:

    st.warning(
        f"Sampling intervals were irregular "
        f"(CV = {dt_cv * 100:.2f}%). "
        "The signal was resampled onto a uniform time grid "
        "before FFT analysis."
    )

else:

    st.success(
        f"Sampling is approximately uniform "
        f"(interval CV = {dt_cv * 100:.4f}%)."
    )


# ============================================================
# SIGNAL PROCESSING
# ============================================================

metrics = calculate_time_metrics(
    accel_data,
    sampling_rate,
)

(
    freqs,
    amplitudes,
    windowed_signal,
) = calculate_fft(
    accel_data,
    sampling_rate,
)


# ============================================================
# DOMINANT PEAKS
# ============================================================

fft_display_limit = min(
    sampling_rate / 2.0,
    max(
        one_x_hz * 15.0,
        500.0,
    ),
)

peak_df = detect_spectral_peaks(
    freqs,
    amplitudes,
    max_frequency=fft_display_limit,
)


# ============================================================
# HARMONIC ANALYSIS
# ============================================================

harmonic_results = {}

for name, target in [
    ("1X", one_x_hz),
    ("2X", two_x_hz),
    ("3X", three_x_hz),
]:

    result = find_harmonic_peak(
        freqs,
        amplitudes,
        target,
    )

    if result:

        harmonic_results[name] = result

    else:

        harmonic_results[name] = None


# ============================================================
# BEARING ANALYSIS
# ============================================================

bearing_frequencies = {}

bearing_matches = {}

if enable_bearing:

    try:

        bearing_frequencies = (
            calculate_bearing_frequencies(
                rpm=rpm,
                number_of_elements=bearing_elements,
                ball_diameter=bearing_ball_diameter,
                pitch_diameter=bearing_pitch_diameter,
                contact_angle_deg=bearing_contact_angle,
            )
        )

        bearing_matches = (
            match_bearing_frequencies(
                peak_df,
                bearing_frequencies,
            )
        )

    except Exception as exc:

        st.warning(
            f"Bearing analysis could not be calculated: "
            f"{exc}"
        )


# ============================================================
# ENVELOPE ANALYSIS
# ============================================================

envelope_figure = None
envelope_freqs = None
envelope_amplitudes = None

if enable_envelope:

    try:

        (
            envelope_freqs,
            envelope_amplitudes,
            envelope_signal,
        ) = calculate_envelope_spectrum(
            accel_data,
            sampling_rate,
            envelope_low,
            envelope_high,
        )

    except Exception as exc:

        st.warning(
            f"Envelope analysis failed: {exc}"
        )


# ============================================================
# METRICS DASHBOARD
# ============================================================

st.header(
    "2️⃣ Vibration Metrics"
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "RMS Acceleration",
    f"{metrics['rms']:.4f} g",
)

c2.metric(
    "Peak Acceleration",
    f"{metrics['peak']:.4f} g",
)

c3.metric(
    "Crest Factor",
    f"{metrics['crest_factor']:.2f}",
)

c4.metric(
    "Kurtosis",
    f"{metrics['kurtosis']:.2f}",
)

if np.isfinite(
    metrics["velocity_rms_mm_s"]
):

    c5.metric(
        "Velocity RMS",
        f"{metrics['velocity_rms_mm_s']:.3f} mm/s",
    )

else:

    c5.metric(
        "Velocity RMS",
        "N/A",
    )


# ============================================================
# VISUALIZATION
# ============================================================

st.header(
    "3️⃣ Vibration Spectrum"
)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    dpi=180,
)


# ------------------------------------------------------------
# TIME DOMAIN
# ------------------------------------------------------------

samples_to_plot = min(
    2000,
    len(time_data),
)

ax1.plot(
    time_data[:samples_to_plot],
    accel_data[:samples_to_plot],
    linewidth=0.9,
)

ax1.set_title(
    "Time-Domain Acceleration Waveform"
)

ax1.set_xlabel(
    "Time (s)"
)

ax1.set_ylabel(
    "Acceleration (g)"
)

ax1.grid(
    True,
    linestyle=":",
    alpha=0.5,
)


# ------------------------------------------------------------
# FFT
# ------------------------------------------------------------

fft_mask = (
    freqs <= fft_display_limit
)

ax2.plot(
    freqs[fft_mask],
    amplitudes[fft_mask],
    linewidth=1.0,
)

ax2.set_xlim(
    0,
    fft_display_limit,
)

ax2.set_title(
    "Single-Sided FFT Amplitude Spectrum"
)

ax2.set_xlabel(
    "Frequency (Hz)"
)

ax2.set_ylabel(
    "Amplitude (g)"
)

ax2.grid(
    True,
    linestyle=":",
    alpha=0.5,
)


# ------------------------------------------------------------
# HARMONIC LINES
# ------------------------------------------------------------

harmonic_styles = [
    ("1X", one_x_hz),
    ("2X", two_x_hz),
    ("3X", three_x_hz),
]

for label, frequency in harmonic_styles:

    if frequency <= fft_display_limit:

        ax2.axvline(
            frequency,
            linestyle="--",
            alpha=0.65,
            label=(
                f"{label} "
                f"({frequency:.2f} Hz)"
            ),
        )


# ------------------------------------------------------------
# BEARING LINES
# ------------------------------------------------------------

if bearing_frequencies:

    for name, frequency in (
        bearing_frequencies.items()
    ):

        if frequency <= fft_display_limit:

            ax2.axvline(
                frequency,
                linestyle=":",
                alpha=0.5,
                label=(
                    f"{name} "
                    f"({frequency:.2f} Hz)"
                ),
            )


ax2.legend(
    loc="upper right",
    fontsize=8,
)


plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True,
)


# ============================================================
# DOMINANT PEAK TABLE
# ============================================================

st.subheader(
    "Dominant Spectral Peaks"
)

if (
    peak_df is not None
    and not peak_df.empty
):

    display_peak_df = peak_df.head(
        15
    ).copy()

    display_peak_df.index = (
        np.arange(
            1,
            len(display_peak_df) + 1,
        )
    )

    display_peak_df.index.name = (
        "Rank"
    )

    st.dataframe(
        display_peak_df,
        use_container_width=True,
    )

else:

    st.info(
        "No significant spectral peaks were detected."
    )


# ============================================================
# HARMONIC TABLE
# ============================================================

st.subheader(
    "1× / 2× / 3× Running-Speed Analysis"
)

harmonic_rows = []

for name in [
    "1X",
    "2X",
    "3X",
]:

    result = harmonic_results.get(
        name
    )

    if result:

        harmonic_rows.append(
            {
                "Component":
                    name,

                "Expected_Hz":
                    result[
                        "expected_frequency"
                    ],

                "Detected_Hz":
                    result[
                        "detected_frequency"
                    ],

                "Amplitude_g":
                    result[
                        "amplitude"
                    ],
            }
        )

    else:

        harmonic_rows.append(
            {
                "Component": name,
                "Expected_Hz": np.nan,
                "Detected_Hz": np.nan,
                "Amplitude_g": np.nan,
            }
        )

st.dataframe(
    pd.DataFrame(
        harmonic_rows
    ),
    use_container_width=True,
)


# ============================================================
# BEARING TABLE
# ============================================================

if bearing_frequencies:

    st.subheader(
        "Bearing Characteristic Frequencies"
    )

    bearing_rows = []

    for name, frequency in (
        bearing_frequencies.items()
    ):

        match = bearing_matches.get(
            name
        )

        bearing_rows.append(
            {
                "Frequency":
                    name,

                "Calculated_Hz":
                    frequency,

                "Matched_Peak_Hz":
                    (
                        match["Frequency_Hz"]
                        if match
                        else np.nan
                    ),

                "Matched_Amplitude_g":
                    (
                        match["Amplitude_g"]
                        if match
                        else np.nan
                    ),
            }
        )

    st.dataframe(
        pd.DataFrame(
            bearing_rows
        ),
        use_container_width=True,
    )


# ============================================================
# ENVELOPE PLOT
# ============================================================

if (
    enable_envelope
    and envelope_freqs is not None
):

    st.subheader(
        "Envelope Spectrum"
    )

    envelope_limit = min(
        envelope_freqs[-1],
        max(
            one_x_hz * 15,
            500,
        ),
    )

    env_mask = (
        envelope_freqs
        <= envelope_limit
    )

    envelope_figure, envelope_ax = (
        plt.subplots(
            figsize=(11, 4),
            dpi=180,
        )
    )

    envelope_ax.plot(
        envelope_freqs[env_mask],
        envelope_amplitudes[env_mask],
        linewidth=1.0,
    )

    envelope_ax.set_xlim(
        0,
        envelope_limit,
    )

    envelope_ax.set_title(
        "Envelope Spectrum"
    )

    envelope_ax.set_xlabel(
        "Frequency (Hz)"
    )

    envelope_ax.set_ylabel(
        "Envelope Amplitude"
    )

    envelope_ax.grid(
        True,
        linestyle=":",
        alpha=0.5,
    )

    plt.tight_layout()

    st.pyplot(
        envelope_figure,
        use_container_width=True,
    )


# ============================================================
# BUILD AI CONTEXT
# ============================================================

analysis_context = build_analysis_context(
    rpm=rpm,
    fs=sampling_rate,
    duration=duration,
    metrics=metrics,
    harmonic_results=harmonic_results,
    peak_df=peak_df,
    bearing_frequencies=bearing_frequencies,
    bearing_matches=bearing_matches,
)

st.session_state.analysis_context = (
    analysis_context
)


# ============================================================
# AI REPORT
# ============================================================

st.header(
    "4️⃣ 🤖 AI Vibration Diagnostic Report"
)

st.markdown(
    """
The AI receives the numerical vibration analysis and the
generated waveform/FFT plot. It should be treated as an
engineering decision-support system rather than a substitute
for qualified vibration analysis.
"""
)


# ------------------------------------------------------------
# RUN AI BUTTON
# ------------------------------------------------------------

if st.button(
    "🔍 Run AI Spectrum Analysis",
    type="primary",
):

    if not api_key.strip():

        st.error(
            "Please enter your Gemini API key in the sidebar."
        )

    else:

        with st.spinner(
            "AI is analyzing the vibration spectrum..."
        ):

            try:

                client = genai.Client(
                    api_key=api_key
                )

                ai_prompt = f"""
You are an experienced rotating-machinery reliability
engineer and vibration analyst.

Analyze the supplied numerical vibration data and the
waveform/FFT plot.

Do not invent measurements.

Clearly distinguish observations from hypotheses.

============================================================
NUMERICAL ANALYSIS
============================================================

{analysis_context}

============================================================
REQUIRED REPORT
============================================================

Produce a professional engineering report with these sections:

1. Executive Summary

2. Data Quality Assessment

3. Time-Domain Signal Profile
   - RMS
   - peak
   - crest factor
   - kurtosis
   - waveform characteristics
   - possible impact/transient behavior

4. Frequency-Domain / FFT Assessment
   - dominant frequencies
   - amplitudes
   - 1X
   - 2X
   - 3X
   - harmonic patterns

5. Bearing-Frequency Assessment
   - BPFO
   - BPFI
   - BSF
   - FTF
   - envelope evidence
   - explicitly state if bearing geometry is unavailable

6. Probable Fault Conditions
   Consider:
   - rotor unbalance
   - shaft/coupling misalignment
   - mechanical looseness
   - bearing defects
   - resonance
   - other plausible conditions

7. Evidence & Confidence
   For each suspected fault give:
   - Observed evidence
   - Engineering interpretation
   - Confidence: High / Moderate / Low

8. Recommended Maintenance / Diagnostic Actions

9. Additional Measurements Required

Important:
A high 1X peak does not automatically prove unbalance.
A high 2X peak does not automatically prove misalignment.
Bearing frequencies should only be considered meaningful when
the bearing geometry and speed support the calculation.

Do not present speculation as a confirmed failure.
"""

                # Convert plot to PNG bytes
                image_buffer = io.BytesIO()

                fig.savefig(
                    image_buffer,
                    format="png",
                    dpi=180,
                    bbox_inches="tight",
                )

                image_bytes = (
                    image_buffer.getvalue()
                )

                image_part = (
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    )
                )

                response = (
                    client.models.generate_content(
                        model=model_name,
                        contents=[
                            ai_prompt,
                            image_part,
                        ],
                    )
                )

                ai_report = response.text

                if not ai_report:

                    raise ValueError(
                        "Gemini returned an empty response."
                    )

                st.session_state.ai_report = (
                    ai_report
                )

                # ------------------------------------------------
                # GENERATE PDF
                # ------------------------------------------------

                pdf_bytes = (
                    generate_pdf_report(
                        ai_report=ai_report,
                        filename=(
                            st.session_state
                            .uploaded_filename
                        ),
                        rpm=rpm,
                        sampling_rate=sampling_rate,
                        duration=duration,
                        num_samples=number_of_samples,
                        rms_accel=metrics[
                            "rms"
                        ],
                        peak_accel=metrics[
                            "peak"
                        ],
                        peak_to_peak=metrics[
                            "peak_to_peak"
                        ],
                        crest_factor=metrics[
                            "crest_factor"
                        ],
                        kurtosis=metrics[
                            "kurtosis"
                        ],
                        skewness=metrics[
                            "skewness"
                        ],
                        velocity_rms=metrics[
                            "velocity_rms_mm_s"
                        ],
                        harmonic_results=(
                            harmonic_results
                        ),
                        peak_df=peak_df,
                        bearing_frequencies=(
                            bearing_frequencies
                        ),
                        bearing_matches=(
                            bearing_matches
                        ),
                        plot_figure=fig,
                        envelope_figure=(
                            envelope_figure
                        ),
                    )
                )

                st.session_state.pdf_report = (
                    pdf_bytes
                )

                st.success(
                    "✅ AI analysis and PDF report "
                    "generated successfully."
                )

            except Exception as exc:

                st.error(
                    "AI analysis failed."
                )

                st.exception(
                    exc
                )


# ============================================================
# DISPLAY AI REPORT
# ============================================================

if st.session_state.ai_report:

    st.markdown(
        st.session_state.ai_report
    )


# ============================================================
# PDF DOWNLOAD
# ============================================================

if st.session_state.pdf_report:

    st.subheader(
        "📄 Diagnostic PDF Report"
    )

    st.success(
        "Your professional vibration diagnostic report "
        "is ready."
    )

    base_name = (
        st.session_state
        .uploaded_filename
        .rsplit(
            ".",
            1,
        )[0]
    )

    pdf_filename = (
        f"{base_name}_"
        f"vibration_diagnostic_report.pdf"
    )

    st.download_button(
        label=(
            "📥 Download Complete Diagnostic Report (PDF)"
        ),
        data=st.session_state.pdf_report,
        file_name=pdf_filename,
        mime="application/pdf",
        type="primary",
    )


# ============================================================
# INTERACTIVE AI ASSISTANT
# ============================================================

st.header(
    "5️⃣ 💬 Interactive Vibration AI Assistant"
)

st.markdown(
    """
Ask questions about the **current vibration dataset**.

Examples:

- Why is the 1X component dominant?
- Does this indicate rotor unbalance?
- What does the crest factor tell me?
- Which frequency has the highest amplitude?
- Could this indicate bearing damage?
- Explain the FFT result in simple terms.
- Why can't BPFO/BPFI be confirmed?
- What should I inspect first?
- What additional measurements should I collect?
"""
)


# ============================================================
# CLEAR CHAT
# ============================================================

if st.session_state.chat_messages:

    if st.button(
        "🗑️ Clear AI Conversation"
    ):

        st.session_state.chat_messages = []

        st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.chat_messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a question about this vibration analysis..."
)


if user_question:

    if not api_key.strip():

        st.error(
            "Please enter your Gemini API key in the sidebar."
        )

    else:

        # ----------------------------------------------------
        # SHOW USER MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_question
            )

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        # ----------------------------------------------------
        # CONVERSATION HISTORY
        # ----------------------------------------------------

        history_text = ""

        for message in (
            st.session_state
            .chat_messages[-12:]
        ):

            role = (
                "USER"
                if message["role"]
                == "user"
                else "AI"
            )

            history_text += (
                f"\n{role}: "
                f"{message['content']}\n"
            )

        # ----------------------------------------------------
        # CHAT PROMPT
        # ----------------------------------------------------

        chat_prompt = f"""
You are an interactive rotating-machinery vibration
diagnostic assistant.

The user is asking a question about the SAME dataset that
was analyzed above.

Use only the supplied analysis data.

============================================================
CURRENT VIBRATION ANALYSIS
============================================================

{st.session_state.analysis_context}

============================================================
PREVIOUS CONVERSATION
============================================================

{history_text}

============================================================
CURRENT QUESTION
============================================================

{user_question}

============================================================
RESPONSE RULES
============================================================

Answer the user's question directly.

Use actual measured values whenever relevant.

Example:

Do not say:
"The 1X frequency is high."

Instead say:
"The calculated 1X running frequency is approximately
29.67 Hz and the FFT should be examined around that
frequency."

Always distinguish:

Observation
What the data actually shows.

Interpretation
What it may indicate.

Confidence
High / Moderate / Low.

Recommended next check
What measurement or inspection would reduce uncertainty.

Never invent:
- sensor readings
- phase information
- bearing geometry
- machine history
- temperatures
- lubrication conditions
- maintenance history

Do not claim that a single frequency proves a fault.

If the available data is insufficient to answer with
confidence, say exactly what additional information is
needed.

Keep the answer focused on the question.
"""

        # ----------------------------------------------------
        # AI RESPONSE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Analyzing your question..."
            ):

                try:

                    client = genai.Client(
                        api_key=api_key
                    )

                    response = (
                        client.models.generate_content(
                            model=model_name,
                            contents=chat_prompt,
                        )
                    )

                    answer = response.text

                    if not answer:

                        raise ValueError(
                            "Gemini returned an empty response."
                        )

                    st.markdown(
                        answer
                    )

                    st.session_state.chat_messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                answer,
                        }
                    )

                except Exception as exc:

                    st.error(
                        f"AI response failed: {exc}"
                    )