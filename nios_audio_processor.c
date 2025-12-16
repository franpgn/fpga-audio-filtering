/*
 * nios_audio_processor.c
 *
 * Unified audio processing for NIOS II:
 * 1. Voice denoising (STFT + Wiener filter)
 * 2. FIR low-pass filtering (applied 3 times)
 *
 * Processing pipeline matches script.sh:
 *   voice_denoise --aggr 1.25 --gmin 0.06
 *   audioFilter 401 12000 (x3)
 *
 * Build for NIOS II:
 *   nios2-elf-gcc -O2 -o nios_audio_processor nios_audio_processor.c -lm
 *
 * Notes:
 *   - Processes 16-bit PCM mono audio
 *   - Sample rate assumed 48000 Hz (configurable)
 *   - Uses floating-point for denoising, Q1.15 fixed-point for FIR
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Configuration - adjust these for your setup
 * ============================================================================ */

#define SAMPLE_RATE       48000
#define FFT_SIZE          1024
#define FFT_HOP           256
#define FFT_BINS          (FFT_SIZE / 2 + 1)

/* FIR filter parameters (matching script: 401 taps, 12kHz cutoff) */
#define FIR_TAPS          401
#define FIR_CUTOFF_HZ     12000
#define FIR_PASSES        3

/* Denoiser parameters (matching script: --aggr 1.25 --gmin 0.06) */
#define DENOISE_AGGR      1.25f
#define DENOISE_GMIN      0.06f
#define DENOISE_VAD_DB    3.0f
#define DENOISE_HP_HZ     100.0f

/* Maximum audio length in samples - adjust based on available memory */
#define MAX_AUDIO_SAMPLES 480000  /* 10 seconds at 48kHz */

/* ============================================================================
 * Static buffers for NIOS II (no dynamic allocation)
 * ============================================================================ */

/* FIR coefficients in Q1.15 format */
static int32_t fir_coeffs[FIR_TAPS];

/* FFT/denoiser working buffers */
static float fft_window[FFT_SIZE];
static float noise_psd[FFT_BINS];
static float xi_state[FFT_BINS];
static float gain_prev[FFT_BINS];
static float ola_buffer[FFT_SIZE];

/* DC blocker state */
static float dc_x1 = 0.0f;
static float dc_y1 = 0.0f;

/* HPF biquad state */
static float hpf_b0, hpf_b1, hpf_b2, hpf_a1, hpf_a2;
static float hpf_z1 = 0.0f, hpf_z2 = 0.0f;

/* Audio processing buffers */
static float audio_float[MAX_AUDIO_SAMPLES];
static float audio_temp[MAX_AUDIO_SAMPLES];

/* ============================================================================
 * Utility functions
 * ============================================================================ */

static inline int16_t sat16(int32_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

static inline int clamp_i(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static inline float clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline float maxf(float a, float b) {
    return (a > b) ? a : b;
}

/* ============================================================================
 * Complex number operations for FFT
 * ============================================================================ */

typedef struct { float re, im; } cplx;

static cplx fft_buf[FFT_SIZE];

static inline cplx c_add(cplx a, cplx b) { return (cplx){a.re + b.re, a.im + b.im}; }
static inline cplx c_sub(cplx a, cplx b) { return (cplx){a.re - b.re, a.im - b.im}; }
static inline cplx c_mul(cplx a, cplx b) { return (cplx){a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re}; }

/* ============================================================================
 * FFT - Radix-2 Cooley-Tukey (in-place)
 * dir = +1 for forward FFT, -1 for inverse FFT
 * ============================================================================ */

static void fft_inplace(cplx *a, int n, int dir) {
    int i, j, bit, len, half;
    cplx tmp, w, wlen, u, v;
    float ang, inv;

    /* Bit-reversal permutation */
    for (i = 1, j = 0; i < n; i++) {
        bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
    }

    /* Butterfly operations */
    for (len = 2; len <= n; len <<= 1) {
        ang = (dir > 0 ? -2.0f : 2.0f) * (float)M_PI / (float)len;
        wlen.re = cosf(ang);
        wlen.im = sinf(ang);
        half = len / 2;

        for (i = 0; i < n; i += len) {
            w.re = 1.0f;
            w.im = 0.0f;
            for (j = 0; j < half; j++) {
                u = a[i + j];
                v = c_mul(a[i + j + half], w);
                a[i + j] = c_add(u, v);
                a[i + j + half] = c_sub(u, v);
                w = c_mul(w, wlen);
            }
        }
    }

    /* Scale for inverse FFT */
    if (dir < 0) {
        inv = 1.0f / (float)n;
        for (i = 0; i < n; i++) {
            a[i].re *= inv;
            a[i].im *= inv;
        }
    }
}

/* ============================================================================
 * Hann window generation
 * ============================================================================ */

static void make_hann_window(float *w, int n) {
    int i;
    for (i = 0; i < n; i++) {
        w[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * (float)i / (float)(n - 1));
    }
}

/* ============================================================================
 * HPF Biquad filter design and processing
 * ============================================================================ */

static void design_hpf_biquad(float fs, float f0, float Q) {
    float w0 = 2.0f * (float)M_PI * f0 / fs;
    float cw = cosf(w0);
    float sw = sinf(w0);
    float alpha = sw / (2.0f * Q);

    float b0 = (1.0f + cw) / 2.0f;
    float b1 = -(1.0f + cw);
    float b2 = (1.0f + cw) / 2.0f;
    float a0 = 1.0f + alpha;
    float a1 = -2.0f * cw;
    float a2 = 1.0f - alpha;

    hpf_b0 = b0 / a0;
    hpf_b1 = b1 / a0;
    hpf_b2 = b2 / a0;
    hpf_a1 = a1 / a0;
    hpf_a2 = a2 / a0;
}

static inline float apply_hpf_biquad(float x) {
    float y = hpf_b0 * x + hpf_z1;
    hpf_z1 = hpf_b1 * x - hpf_a1 * y + hpf_z2;
    hpf_z2 = hpf_b2 * x - hpf_a2 * y;
    return y;
}

/* ============================================================================
 * DC Blocker: y[n] = x[n] - x[n-1] + R*y[n-1]
 * ============================================================================ */

static inline float apply_dc_block(float x) {
    const float R = 0.995f;
    float y = x - dc_x1 + R * dc_y1;
    dc_x1 = x;
    dc_y1 = y;
    return y;
}

static void reset_dc_block(void) {
    dc_x1 = 0.0f;
    dc_y1 = 0.0f;
}

static void reset_hpf(void) {
    hpf_z1 = 0.0f;
    hpf_z2 = 0.0f;
}

/* ============================================================================
 * VAD (Voice Activity Detection) - simple energy-based
 * ============================================================================ */

static int vad_is_speech(const float *psd, int bins, float vad_db) {
    double P = 0.0, N = 0.0;
    int k;
    double snr, snr_db;

    for (k = 1; k < bins; k++) {
        P += psd[k];
        N += noise_psd[k] + 1e-12;
    }

    snr = P / N;
    snr_db = 10.0 * log10(snr + 1e-12);

    return snr_db > vad_db;
}

/* ============================================================================
 * Voice Denoiser - STFT + Wiener gain
 * ============================================================================ */

static void init_denoiser(void) {
    int k;

    make_hann_window(fft_window, FFT_SIZE);

    for (k = 0; k < FFT_BINS; k++) {
        noise_psd[k] = 0.0f;
        xi_state[k] = 0.0f;
        gain_prev[k] = 1.0f;
    }

    for (k = 0; k < FFT_SIZE; k++) {
        ola_buffer[k] = 0.0f;
    }

    /* Initialize pre-filters */
    reset_dc_block();
    design_hpf_biquad((float)SAMPLE_RATE, DENOISE_HP_HZ, 0.707f);
    reset_hpf();
}

static void denoise_process(const float *in, float *out, int n_samples) {
    const float noise_alpha = 0.98f;
    const float xi_alpha = 0.90f;
    const float gain_smooth = 0.80f;
    const float gmin = DENOISE_GMIN;
    const float aggr = DENOISE_AGGR;
    const float vad_db = DENOISE_VAD_DB;

    int init_noise_frames = (int)ceilf((0.35f * (float)SAMPLE_RATE) / (float)FFT_HOP);
    if (init_noise_frames < 2) init_noise_frames = 2;

    float psd[FFT_BINS];
    int frame_idx = 0;
    int start, n, k, idx, km;
    int speech;
    float x, re, im, Nk, Pk, gamma, gamma_m1, xi_new, G, y;

    /* Clear output */
    for (n = 0; n < n_samples; n++) {
        out[n] = 0.0f;
    }

    /* Reset OLA buffer */
    for (n = 0; n < FFT_SIZE; n++) {
        ola_buffer[n] = 0.0f;
    }

    /* Process frame by frame */
    for (start = 0; start < n_samples; start += FFT_HOP, frame_idx++) {

        /* Analysis: window input samples */
        for (n = 0; n < FFT_SIZE; n++) {
            idx = start + n;
            x = (idx < n_samples) ? in[idx] : 0.0f;
            fft_buf[n].re = x * fft_window[n];
            fft_buf[n].im = 0.0f;
        }

        /* Forward FFT */
        fft_inplace(fft_buf, FFT_SIZE, +1);

        /* Compute PSD for positive frequencies */
        for (k = 0; k < FFT_BINS; k++) {
            re = fft_buf[k].re;
            im = fft_buf[k].im;
            psd[k] = re * re + im * im;
        }

        /* VAD and noise estimation */
        if (frame_idx < init_noise_frames) {
            speech = 0;  /* Bootstrap: assume noise */
        } else {
            speech = vad_is_speech(psd, FFT_BINS, vad_db);
        }

        /* Update noise estimate during non-speech */
        if (!speech) {
            for (k = 0; k < FFT_BINS; k++) {
                Nk = noise_psd[k];
                Pk = psd[k];
                noise_psd[k] = noise_alpha * Nk + (1.0f - noise_alpha) * Pk;
            }
        }

        /* Compute and apply Wiener gain */
        for (k = 0; k < FFT_BINS; k++) {
            Nk = noise_psd[k] + 1e-12f;
            gamma = psd[k] / Nk;  /* a-posteriori SNR */
            gamma_m1 = maxf(gamma - 1.0f, 0.0f);

            /* Decision-directed a-priori SNR */
            xi_new = xi_alpha * xi_state[k] + (1.0f - xi_alpha) * gamma_m1;
            xi_new = xi_new / aggr;  /* Aggressiveness control */

            /* Wiener gain */
            G = xi_new / (1.0f + xi_new);

            /* Gain floor and ceiling */
            if (G < gmin) G = gmin;
            if (G > 1.0f) G = 1.0f;

            /* Temporal smoothing */
            G = gain_smooth * gain_prev[k] + (1.0f - gain_smooth) * G;

            xi_state[k] = xi_new;
            gain_prev[k] = G;

            /* Apply gain to positive frequency bin */
            fft_buf[k].re *= G;
            fft_buf[k].im *= G;

            /* Apply to corresponding negative frequency (conjugate symmetry) */
            if (k > 0 && k < FFT_BINS - 1) {
                km = FFT_SIZE - k;
                fft_buf[km].re *= G;
                fft_buf[km].im *= G;
            }
        }

        /* Inverse FFT */
        fft_inplace(fft_buf, FFT_SIZE, -1);

        /* Synthesis: window and overlap-add */
        for (n = 0; n < FFT_SIZE; n++) {
            y = fft_buf[n].re * fft_window[n];
            ola_buffer[n] += y;
        }

        /* Output hop samples */
        for (n = 0; n < FFT_HOP; n++) {
            idx = start + n;
            if (idx < n_samples) {
                out[idx] = ola_buffer[n];
            }
        }

        /* Shift OLA buffer */
        memmove(ola_buffer, ola_buffer + FFT_HOP, (FFT_SIZE - FFT_HOP) * sizeof(float));
        memset(ola_buffer + (FFT_SIZE - FFT_HOP), 0, FFT_HOP * sizeof(float));
    }
}

/* ============================================================================
 * FIR Low-Pass Filter Design (Windowed-Sinc with Hamming)
 * Coefficients stored in Q1.15 fixed-point format
 * ============================================================================ */

static void design_fir_lowpass_q15(int32_t *hq15, int taps, float cutoff_hz, uint32_t sample_rate) {
    int mid = taps / 2;
    float fc = cutoff_hz / (float)sample_rate;
    float h[FIR_TAPS];
    float sum = 0.0f;
    int n, k;
    float x, a, w;
    int64_t SCALE = (int64_t)1 << 15;
    int64_t sum_q = 0;
    int64_t q, delta;

    /* Compute ideal sinc low-pass with Hamming window */
    for (n = 0; n < taps; n++) {
        k = n - mid;
        if (k == 0) {
            x = 2.0f * fc;
        } else {
            a = 2.0f * (float)M_PI * fc * (float)k;
            x = sinf(a) / ((float)M_PI * (float)k);
        }

        /* Hamming window */
        w = 0.54f - 0.46f * cosf((2.0f * (float)M_PI * (float)n) / (float)(taps - 1));
        h[n] = x * w;
    }

    /* Normalize for unity DC gain */
    for (n = 0; n < taps; n++) {
        sum += h[n];
    }
    for (n = 0; n < taps; n++) {
        h[n] /= sum;
    }

    /* Quantize to Q1.15 */
    for (n = 0; n < taps; n++) {
        q = (int64_t)lrintf(h[n] * (float)SCALE);
        hq15[n] = (int32_t)q;
        sum_q += q;
    }

    /* Adjust center tap for exact unity gain */
    delta = SCALE - sum_q;
    hq15[mid] = (int32_t)((int64_t)hq15[mid] + delta);
}

/* ============================================================================
 * FIR Filter Application (Q1.15 fixed-point)
 * ============================================================================ */

static void apply_fir_filter_q15(const int16_t *in, int16_t *out, int n_samples,
                                  const int32_t *hq15, int taps) {
    int mid = taps / 2;
    int64_t ROUND = (int64_t)1 << 14;
    int i, k, idx;
    int64_t acc;
    int32_t y;

    for (i = 0; i < n_samples; i++) {
        acc = 0;

        /* MAC loop */
        for (k = 0; k < taps; k++) {
            idx = clamp_i(i + k - mid, 0, n_samples - 1);
            acc += (int64_t)hq15[k] * (int64_t)in[idx];
        }

        /* Q15 to int16 with rounding */
        y = (int32_t)((acc + ROUND) >> 15);
        out[i] = sat16(y);
    }
}

/* ============================================================================
 * Main Processing Pipeline
 * ============================================================================ */

/**
 * Initialize the audio processor.
 * Call this once before processing audio.
 */
void audio_processor_init(void) {
    /* Initialize denoiser */
    init_denoiser();

    /* Design FIR filter coefficients */
    design_fir_lowpass_q15(fir_coeffs, FIR_TAPS, FIR_CUTOFF_HZ, SAMPLE_RATE);
}

/**
 * Process audio samples through the complete pipeline:
 * 1. Pre-filtering (DC block + HPF)
 * 2. Voice denoising (STFT + Wiener)
 * 3. FIR low-pass filtering (3 passes)
 *
 * @param input   Input audio samples (16-bit PCM)
 * @param output  Output audio samples (16-bit PCM)
 * @param n_samples Number of samples to process
 */
void audio_processor_process(const int16_t *input, int16_t *output, int n_samples) {
    int i;
    float x;
    int pass;

    if (n_samples > MAX_AUDIO_SAMPLES) {
        n_samples = MAX_AUDIO_SAMPLES;
    }

    /* Reset filter states for each new audio block */
    reset_dc_block();
    reset_hpf();
    init_denoiser();

    /* ========== Stage 1: Pre-filtering + Denoising ========== */

    /* Convert to float and apply pre-filters (DC block + HPF) */
    for (i = 0; i < n_samples; i++) {
        x = (float)input[i] / 32768.0f;
        x = apply_dc_block(x);
        x = apply_hpf_biquad(x);
        audio_float[i] = x;
    }

    /* Apply STFT-based denoising */
    denoise_process(audio_float, audio_temp, n_samples);

    /* Convert back to int16 */
    for (i = 0; i < n_samples; i++) {
        float y = clampf(audio_temp[i], -1.0f, 1.0f);
        int32_t si = (int32_t)lrintf(y * 32767.0f);
        output[i] = sat16(si);
    }

    /* ========== Stage 2: FIR Low-Pass Filtering (3 passes) ========== */

    for (pass = 0; pass < FIR_PASSES; pass++) {
        /* Use a temporary buffer approach: read from output, write back to output */
        /* First copy output to a temp buffer (reuse audio_float as int16 temp) */
        int16_t *temp_in = (int16_t *)audio_float;  /* Reuse buffer */
        memcpy(temp_in, output, n_samples * sizeof(int16_t));

        /* Apply FIR filter */
        apply_fir_filter_q15(temp_in, output, n_samples, fir_coeffs, FIR_TAPS);
    }
}

/**
 * Process a single sample (for real-time streaming).
 * Note: For best quality, use block processing with audio_processor_process().
 *
 * This is a simplified version that only applies the pre-filtering and FIR.
 * STFT denoising requires block processing.
 */
int16_t audio_processor_process_sample(int16_t input) {
    /* For real-time single-sample processing, only pre-filtering is practical */
    /* STFT denoising requires block processing */
    float x = (float)input / 32768.0f;
    x = apply_dc_block(x);
    x = apply_hpf_biquad(x);
    x = clampf(x, -1.0f, 1.0f);
    return sat16((int32_t)lrintf(x * 32767.0f));
}

/* ============================================================================
 * Test/Demo Main (can be disabled for embedded use)
 * ============================================================================ */

#ifndef NIOS_EMBEDDED

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* WAV file handling for testing */

typedef struct {
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} WavFmt;

typedef struct {
    WavFmt fmt;
    uint32_t data_size;
    long data_offset;
} WavInfo;

static void die(const char *msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(EXIT_FAILURE);
}

static void die_errno(const char *context) {
    fprintf(stderr, "Error: %s: %s\n", context, strerror(errno));
    exit(EXIT_FAILURE);
}

static uint16_t le16(const uint8_t b[2]) {
    return (uint16_t)(b[0] | ((uint16_t)b[1] << 8));
}

static uint32_t le32(const uint8_t b[4]) {
    return (uint32_t)(b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24));
}

static void put_le16(uint8_t b[2], uint16_t v) {
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
}

static void put_le32(uint8_t b[4], uint32_t v) {
    b[0] = (uint8_t)(v & 0xFF);
    b[1] = (uint8_t)((v >> 8) & 0xFF);
    b[2] = (uint8_t)((v >> 16) & 0xFF);
    b[3] = (uint8_t)((v >> 24) & 0xFF);
}

static int is_fourcc(const uint8_t id[4], const char *s) {
    return id[0]==(uint8_t)s[0] && id[1]==(uint8_t)s[1] && id[2]==(uint8_t)s[2] && id[3]==(uint8_t)s[3];
}

static WavInfo parse_wav(FILE *in) {
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, in) != 12) die("Failed to read WAV header");
    if (!is_fourcc(&hdr[0], "RIFF")) die("Not a RIFF file");
    if (!is_fourcc(&hdr[8], "WAVE")) die("Not a WAVE file");

    WavInfo info;
    memset(&info, 0, sizeof(info));

    int have_fmt = 0, have_data = 0;

    while (!have_data) {
        uint8_t chunk_hdr[8];
        if (fread(chunk_hdr, 1, 8, in) != 8) die("EOF before data chunk");

        uint8_t *ck_id = &chunk_hdr[0];
        uint32_t ck_size = le32(&chunk_hdr[4]);

        if (is_fourcc(ck_id, "fmt ")) {
            if (ck_size < 16) die("Invalid fmt chunk");
            uint8_t fmtbuf[16];
            if (fread(fmtbuf, 1, 16, in) != 16) die("Failed to read fmt");

            info.fmt.audio_format = le16(&fmtbuf[0]);
            info.fmt.num_channels = le16(&fmtbuf[2]);
            info.fmt.sample_rate = le32(&fmtbuf[4]);
            info.fmt.byte_rate = le32(&fmtbuf[8]);
            info.fmt.block_align = le16(&fmtbuf[12]);
            info.fmt.bits_per_sample = le16(&fmtbuf[14]);

            if (ck_size > 16) fseek(in, ck_size - 16, SEEK_CUR);
            have_fmt = 1;
        } else if (is_fourcc(ck_id, "data")) {
            info.data_size = ck_size;
            info.data_offset = ftell(in);
            have_data = 1;
        } else {
            fseek(in, ck_size, SEEK_CUR);
        }
        if (ck_size & 1) fseek(in, 1, SEEK_CUR);
    }

    if (!have_fmt) die("Missing fmt chunk");
    if (info.fmt.audio_format != 1) die("Only PCM supported");
    if (info.fmt.bits_per_sample != 16) die("Only 16-bit supported");
    if (info.fmt.num_channels != 1) die("Only mono supported for NIOS II version");

    return info;
}

static void write_wav_header(FILE *out, const WavFmt *fmt, uint32_t data_size) {
    uint8_t hdr[44];

    memcpy(&hdr[0], "RIFF", 4);
    put_le32(&hdr[4], 36 + data_size);
    memcpy(&hdr[8], "WAVE", 4);
    memcpy(&hdr[12], "fmt ", 4);
    put_le32(&hdr[16], 16);
    put_le16(&hdr[20], fmt->audio_format);
    put_le16(&hdr[22], fmt->num_channels);
    put_le32(&hdr[24], fmt->sample_rate);
    put_le32(&hdr[28], fmt->byte_rate);
    put_le16(&hdr[32], fmt->block_align);
    put_le16(&hdr[34], fmt->bits_per_sample);
    memcpy(&hdr[36], "data", 4);
    put_le32(&hdr[40], data_size);

    fwrite(hdr, 1, 44, out);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.wav> <output.wav>\n", argv[0]);
        fprintf(stderr, "\nProcesses mono 16-bit WAV through:\n");
        fprintf(stderr, "  1. Voice denoising (STFT + Wiener, aggr=%.2f, gmin=%.2f)\n", DENOISE_AGGR, DENOISE_GMIN);
        fprintf(stderr, "  2. FIR low-pass filter (%d taps, %d Hz cutoff, %d passes)\n", FIR_TAPS, FIR_CUTOFF_HZ, FIR_PASSES);
        return 1;
    }

    const char *in_path = argv[1];
    const char *out_path = argv[2];

    FILE *in = fopen(in_path, "rb");
    if (!in) die_errno("fopen(input)");

    WavInfo info = parse_wav(in);
    fseek(in, info.data_offset, SEEK_SET);

    int n_samples = info.data_size / 2;
    if (n_samples > MAX_AUDIO_SAMPLES) {
        fprintf(stderr, "Warning: Truncating to %d samples\n", MAX_AUDIO_SAMPLES);
        n_samples = MAX_AUDIO_SAMPLES;
    }

    int16_t *pcm_in = (int16_t *)malloc(n_samples * sizeof(int16_t));
    int16_t *pcm_out = (int16_t *)malloc(n_samples * sizeof(int16_t));
    if (!pcm_in || !pcm_out) die("Out of memory");

    if (fread(pcm_in, sizeof(int16_t), n_samples, in) != (size_t)n_samples) {
        die("Failed to read samples");
    }
    fclose(in);

    printf("Processing %d samples (%.2f seconds)...\n", n_samples, (float)n_samples / SAMPLE_RATE);

    /* Initialize and process */
    audio_processor_init();
    audio_processor_process(pcm_in, pcm_out, n_samples);

    /* Write output */
    FILE *out = fopen(out_path, "wb");
    if (!out) die_errno("fopen(output)");

    write_wav_header(out, &info.fmt, n_samples * 2);
    fwrite(pcm_out, sizeof(int16_t), n_samples, out);
    fclose(out);

    printf("Done. Output written to %s\n", out_path);

    free(pcm_in);
    free(pcm_out);
    return 0;
}

#endif /* NIOS_EMBEDDED */
