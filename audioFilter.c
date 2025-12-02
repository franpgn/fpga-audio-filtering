/*
 * audioFilter.c
 *
 * FIR low-pass filter for 16-bit PCM WAV files (mono/stereo).
 * - Supports little-endian RIFF/WAVE.
 * - Handles "fmt " and "data" chunks (skips unknown chunks).
 * - Designs a linear-phase low-pass FIR using windowed-sinc (Hamming window).
 * - Applies FIR per channel with edge clamping (same length, no sample shift).
 * - Coefficients are quantized to Q1.15 and MAC uses int64 accumulator (FPGA-friendly).
 *
 * Build:
 *   gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -o audioFilter audioFilter.c -lm
 *
 * Run:
 *   ./audioFilter input.wav output.wav <taps> [cutoff_hz]
 *
 * Notes:
 *   - Only 16-bit PCM supported (AudioFormat=1, BitsPerSample=16).
 *   - taps must be odd and >= 3.
 *   - cutoff_hz must be > 0 and < Nyquist (sample_rate/2).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#if defined(_MSC_VER)
  #define INLINE __inline
#else
  #define INLINE inline
#endif

#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

typedef struct {
  uint16_t audio_format;     // 1 = PCM
  uint16_t num_channels;     // 1 or 2 typically
  uint32_t sample_rate;
  uint32_t byte_rate;
  uint16_t block_align;
  uint16_t bits_per_sample;  // must be 16
} WavFmt;

typedef struct {
  WavFmt fmt;
  uint32_t data_size;   // bytes
  long data_offset;     // file offset where data begins
} WavInfo;

static void die(const char *msg) {
  fprintf(stderr, "Error: %s\n", msg);
  exit(EXIT_FAILURE);
}

static void die_errno(const char *context) {
  fprintf(stderr, "Error: %s: %s\n", context, strerror(errno));
  exit(EXIT_FAILURE);
}

static int read_exact(FILE *f, void *buf, size_t n) {
  return fread(buf, 1, n, f) == n;
}

static int write_exact(FILE *f, const void *buf, size_t n) {
  return fwrite(buf, 1, n, f) == n;
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

static void skip_bytes(FILE *f, uint32_t n) {
  if (fseek(f, (long)n, SEEK_CUR) != 0) die_errno("fseek(skip)");
}

static WavInfo parse_wav(FILE *in) {
  uint8_t hdr[12];
  if (!read_exact(in, hdr, sizeof(hdr))) die("Failed to read WAV header");
  if (!is_fourcc(&hdr[0], "RIFF")) die("Not a RIFF file");
  if (!is_fourcc(&hdr[8], "WAVE")) die("Not a WAVE file");

  WavInfo info;
  memset(&info, 0, sizeof(info));

  int have_fmt = 0;
  int have_data = 0;

  while (!have_data) {
    uint8_t chunk_hdr[8];
    if (!read_exact(in, chunk_hdr, sizeof(chunk_hdr))) {
      die("Reached EOF before finding data chunk");
    }

    uint8_t *ck_id = &chunk_hdr[0];
    uint32_t ck_size = le32(&chunk_hdr[4]);

    if (is_fourcc(ck_id, "fmt ")) {
      if (ck_size < 16) die("Invalid fmt chunk size");

      uint8_t fmtbuf[16];
      if (!read_exact(in, fmtbuf, sizeof(fmtbuf))) die("Failed to read fmt chunk");

      info.fmt.audio_format   = le16(&fmtbuf[0]);
      info.fmt.num_channels   = le16(&fmtbuf[2]);
      info.fmt.sample_rate    = le32(&fmtbuf[4]);
      info.fmt.byte_rate      = le32(&fmtbuf[8]);
      info.fmt.block_align    = le16(&fmtbuf[12]);
      info.fmt.bits_per_sample= le16(&fmtbuf[14]);

      if (ck_size > 16) skip_bytes(in, ck_size - 16);
      have_fmt = 1;

    } else if (is_fourcc(ck_id, "data")) {
      info.data_size = ck_size;
      info.data_offset = ftell(in);
      if (info.data_offset < 0) die_errno("ftell(data_offset)");
      have_data = 1;

    } else {
      skip_bytes(in, ck_size);
    }

    // RIFF chunks are word-aligned (pad 1 byte if size is odd)
    if (ck_size & 1u) skip_bytes(in, 1);
  }

  if (!have_fmt) die("Missing fmt chunk");
  if (!have_data) die("Missing data chunk");

  if (info.fmt.audio_format != 1) die("Only PCM (audio_format=1) is supported");
  if (info.fmt.bits_per_sample != 16) die("Only 16-bit PCM WAV is supported");

  if (info.fmt.num_channels == 0) die("Invalid channel count");
  if (info.fmt.block_align != (uint16_t)(info.fmt.num_channels * 2)) {
    die("Unexpected block_align; expected num_channels * 2 for 16-bit audio");
  }

  return info;
}

static void write_wav_header(FILE *out, const WavFmt *fmt, uint32_t data_size) {
  const uint32_t fmt_size = 16;
  uint32_t riff_size = 4 + (8 + fmt_size) + (8 + data_size);

  uint8_t hdr[12];
  memcpy(&hdr[0], "RIFF", 4);
  put_le32(&hdr[4], riff_size);
  memcpy(&hdr[8], "WAVE", 4);
  if (!write_exact(out, hdr, sizeof(hdr))) die("Failed to write RIFF header");

  uint8_t fmt_hdr[8];
  memcpy(&fmt_hdr[0], "fmt ", 4);
  put_le32(&fmt_hdr[4], fmt_size);
  if (!write_exact(out, fmt_hdr, sizeof(fmt_hdr))) die("Failed to write fmt header");

  uint8_t fmtbuf[16];
  put_le16(&fmtbuf[0],  fmt->audio_format);
  put_le16(&fmtbuf[2],  fmt->num_channels);
  put_le32(&fmtbuf[4],  fmt->sample_rate);
  put_le32(&fmtbuf[8],  fmt->byte_rate);
  put_le16(&fmtbuf[12], fmt->block_align);
  put_le16(&fmtbuf[14], fmt->bits_per_sample);
  if (!write_exact(out, fmtbuf, sizeof(fmtbuf))) die("Failed to write fmt body");

  uint8_t data_hdr[8];
  memcpy(&data_hdr[0], "data", 4);
  put_le32(&data_hdr[4], data_size);
  if (!write_exact(out, data_hdr, sizeof(data_hdr))) die("Failed to write data header");
}

static INLINE int clamp_i(int v, int lo, int hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}

static INLINE int16_t sat16(int32_t x) {
  if (x > 32767) return 32767;
  if (x < -32768) return -32768;
  return (int16_t)x;
}

static void usage(const char *argv0) {
  fprintf(stderr,
    "Usage:\n"
    "  %s <input.wav> <output.wav> <taps> [cutoff_hz]\n\n"
    "Where:\n"
    "  <taps>      odd integer >= 3 (e.g., 31, 51, 101)\n"
    "  [cutoff_hz] optional low-pass cutoff in Hz (must be < sample_rate/2)\n"
    "\nExamples:\n"
    "  %s in.wav out.wav 101 8000\n"
    "  %s in.wav out.wav 51\n",
    argv0, argv0, argv0
  );
}

/*
 * FIR design: windowed-sinc low-pass with Hamming window.
 * Output coefficients are Q1.15 integers (scale = 2^15).
 *
 * Steps:
 * 1) build double h[n] from ideal sinc low-pass
 * 2) apply Hamming window
 * 3) normalize DC gain (sum = 1.0)
 * 4) quantize to Q1.15
 * 5) adjust center tap so sum_q15 == 2^15 (exact unity gain in fixed-point)
 */
static void fir_design_lowpass_q15(int32_t *hq15, int taps, double cutoff_hz, uint32_t sample_rate) {
  if (taps < 3 || (taps % 2) == 0) die("taps must be odd and >= 3");
  if (!(cutoff_hz > 0.0)) die("cutoff_hz must be > 0");
  const double nyquist = (double)sample_rate * 0.5;
  if (!(cutoff_hz < nyquist)) die("cutoff_hz must be < sample_rate/2");

  const int mid = taps / 2;
  const double fc = cutoff_hz / (double)sample_rate; // normalized (0..0.5)
  double *h = (double *)calloc((size_t)taps, sizeof(double));
  if (!h) die("Out of memory (fir coeffs)");

  // Ideal low-pass impulse response (sinc), then window it
  for (int n = 0; n < taps; n++) {
    const int k = n - mid;
    double x;
    if (k == 0) {
      x = 2.0 * fc;
    } else {
      const double a = 2.0 * M_PI * fc * (double)k;
      x = sin(a) / (M_PI * (double)k);
    }

    // Hamming window: 0.54 - 0.46 cos(2*pi*n/(taps-1))
    const double w = 0.54 - 0.46 * cos((2.0 * M_PI * (double)n) / (double)(taps - 1));
    h[n] = x * w;
  }

  // Normalize (unity DC gain)
  double sum = 0.0;
  for (int n = 0; n < taps; n++) sum += h[n];
  if (fabs(sum) < 1e-12) die("FIR design failed (sum ~ 0)");

  for (int n = 0; n < taps; n++) h[n] /= sum;

  // Quantize to Q1.15
  // We use 2^15 scaling (not 2^15-1) to make unity sum adjustment cleaner.
  const int64_t SCALE = (int64_t)1 << 15; // 32768
  int64_t sum_q = 0;
  for (int n = 0; n < taps; n++) {
    // round to nearest
    int64_t q = (int64_t)llround(h[n] * (double)SCALE);
    hq15[n] = (int32_t)q;
    sum_q += q;
  }

  // Adjust center tap so fixed-point coefficients sum exactly to SCALE
  const int64_t delta = SCALE - sum_q;
  hq15[mid] = (int32_t)((int64_t)hq15[mid] + delta);

  free(h);
}

/*
 * Apply FIR per channel, with centered convolution and edge clamping,
 * preserving length and avoiding time shift (offline processing behavior).
 *
 * y[i] = sum_{k=0..taps-1} h[k] * x[ clamp(i + k - mid) ]
 *
 * h is Q1.15, x is int16, acc uses int64:
 * acc_q15 = sum( hq15[k] * x )
 * y = round(acc_q15 / 2^15)
 */
static void fir_filter_interleaved_centered_q15(
    const int16_t *in,
    int16_t *out,
    int frames,
    int channels,
    const int32_t *hq15,
    int taps
) {
  const int mid = taps / 2;
  const int64_t ROUND = (int64_t)1 << 14; // for rounding before >>15

  for (int ch = 0; ch < channels; ch++) {
    for (int i = 0; i < frames; i++) {
      int64_t acc = 0;

      // MAC loop (THIS is the critical part for VHDL)
      for (int k = 0; k < taps; k++) {
        int idx = clamp_i(i + k - mid, 0, frames - 1);
        int16_t x = in[idx * channels + ch];
        acc += (int64_t)hq15[k] * (int64_t)x;
      }

      // Q15 -> int16 with rounding
      int32_t y = (int32_t)((acc + ROUND) >> 15);
      out[i * channels + ch] = sat16(y);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    usage(argv[0]);
    return EXIT_FAILURE;
  }

  const char *in_path  = argv[1];
  const char *out_path = argv[2];

  char *endp = NULL;
  long taps_l = strtol(argv[3], &endp, 10);
  if (!endp || *endp != '\0') die("Invalid taps argument");
  if (taps_l < 3 || (taps_l % 2) == 0 || taps_l > 8191) {
    die("taps must be odd, >= 3, and <= 8191");
  }
  const int taps = (int)taps_l;

  FILE *in = fopen(in_path, "rb");
  if (!in) die_errno("fopen(input)");
  WavInfo info = parse_wav(in);

  // Determine cutoff
  double cutoff_hz;
  if (argc == 5) {
    char *endc = NULL;
    cutoff_hz = strtod(argv[4], &endc);
    if (!endc || *endc != '\0') die("Invalid cutoff_hz argument");
  } else {
    // Default cutoff:
    // 0.225*Fs (~0.45*Nyquist), capped at 8000 Hz, and at least 200 Hz.
    const double fs = (double)info.fmt.sample_rate;
    const double nyq = fs * 0.5;
    cutoff_hz = 0.225 * fs;
    if (cutoff_hz > 8000.0) cutoff_hz = 8000.0;
    if (cutoff_hz < 200.0)  cutoff_hz = 200.0;
    if (cutoff_hz >= nyq) cutoff_hz = nyq * 0.49;
  }

  // Seek to data
  if (fseek(in, info.data_offset, SEEK_SET) != 0) die_errno("fseek(data_offset)");

  const int bytes_per_sample = info.fmt.bits_per_sample / 8; // 2
  const int channels = (int)info.fmt.num_channels;
  const int frame_size = channels * bytes_per_sample;

  if (frame_size <= 0) die("Invalid frame size");
  if (info.data_size % (uint32_t)frame_size != 0) {
    die("Data chunk size is not a multiple of frame size (corrupt or unsupported)");
  }

  const int frames = (int)(info.data_size / (uint32_t)frame_size);

  // Allocate buffers
  const size_t want = (size_t)frames * (size_t)channels;
  int16_t *samples_in  = (int16_t *)malloc(want * sizeof(int16_t));
  int16_t *samples_out = (int16_t *)malloc(want * sizeof(int16_t));
  if (!samples_in || !samples_out) die("Out of memory (audio buffers)");

  if (fread(samples_in, sizeof(int16_t), want, in) != want) {
    die("Failed to read PCM samples (expected 16-bit samples)");
  }
  fclose(in);

  // Design FIR coefficients in Q1.15
  int32_t *hq15 = (int32_t *)malloc((size_t)taps * sizeof(int32_t));
  if (!hq15) die("Out of memory (fir coefficients)");
  fir_design_lowpass_q15(hq15, taps, cutoff_hz, info.fmt.sample_rate);

  // Apply FIR
  fir_filter_interleaved_centered_q15(samples_in, samples_out, frames, channels, hq15, taps);

  // Write output WAV
  FILE *out = fopen(out_path, "wb");
  if (!out) die_errno("fopen(output)");

  WavFmt outfmt = info.fmt;
  outfmt.block_align = (uint16_t)(outfmt.num_channels * 2);
  outfmt.byte_rate = outfmt.sample_rate * (uint32_t)outfmt.block_align;

  write_wav_header(out, &outfmt, info.data_size);

  if (fwrite(samples_out, sizeof(int16_t), want, out) != want) {
    die("Failed to write output samples");
  }

  fclose(out);
  free(hq15);
  free(samples_in);
  free(samples_out);

  return EXIT_SUCCESS;
}
