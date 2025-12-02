/*
 * voice_denoise.c
 *
 * Speech enhancement clássico (sem ML) para WAV 16-bit PCM (mono/stereo):
 * - DC blocker + HPF biquad (+ notch opcional)
 * - STFT (Hann), VAD simples, estimação de ruído por bin
 * - Ganho Wiener com decisão-dirigida + smoothing + gain floor
 * - iSTFT + overlap-add, saída 16-bit
 *
 * Recomendado p/ voz:
 *   N=1024, hop=256 (75% overlap), Fs=48k ou 44.1k
 *
 * Build:
 *   gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -o voice_denoise voice_denoise.c -lm
 *
 * Run:
 *   ./voice_denoise in.wav out.wav [--N 1024] [--hop 256] [--hp 100] [--notch 0|50|60]
 *                             [--gmin 0.08] [--aggr 1.0] [--vad_db 3.0]
 *
 * Observação: sem microfone direcional/array e sem ML, NÃO “isola perfeitamente”
 * outra voz de fundo. Mas reduz bastante ruído estacionário e melhora inteligibilidade.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------- WAV parsing (do seu código, mantido) -----------------

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
    if (!read_exact(in, chunk_hdr, sizeof(chunk_hdr))) die("EOF before data chunk");
    uint8_t *ck_id = &chunk_hdr[0];
    uint32_t ck_size = le32(&chunk_hdr[4]);

    if (is_fourcc(ck_id, "fmt ")) {
      if (ck_size < 16) die("Invalid fmt chunk size");
      uint8_t fmtbuf[16];
      if (!read_exact(in, fmtbuf, sizeof(fmtbuf))) die("Failed to read fmt chunk");

      info.fmt.audio_format    = le16(&fmtbuf[0]);
      info.fmt.num_channels    = le16(&fmtbuf[2]);
      info.fmt.sample_rate     = le32(&fmtbuf[4]);
      info.fmt.byte_rate       = le32(&fmtbuf[8]);
      info.fmt.block_align     = le16(&fmtbuf[12]);
      info.fmt.bits_per_sample = le16(&fmtbuf[14]);

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

    // word-aligned
    if (ck_size & 1u) skip_bytes(in, 1);
  }

  if (!have_fmt) die("Missing fmt chunk");
  if (!have_data) die("Missing data chunk");
  if (info.fmt.audio_format != 1) die("Only PCM supported");
  if (info.fmt.bits_per_sample != 16) die("Only 16-bit PCM supported");
  if (info.fmt.num_channels == 0) die("Invalid channel count");
  if (info.fmt.block_align != (uint16_t)(info.fmt.num_channels * 2)) die("Unexpected block_align");

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

// ----------------- DSP utils -----------------

static inline int16_t sat16i(int32_t x) {
  if (x > 32767) return 32767;
  if (x < -32768) return -32768;
  return (int16_t)x;
}

static inline float clampf(float x, float lo, float hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}

typedef struct { float re, im; } cplx;

static inline cplx c_add(cplx a, cplx b){ return (cplx){a.re+b.re, a.im+b.im}; }
static inline cplx c_sub(cplx a, cplx b){ return (cplx){a.re-b.re, a.im-b.im}; }
static inline cplx c_mul(cplx a, cplx b){ return (cplx){a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re}; }

// FFT radix-2 iterative (in-place). dir=+1 FFT, dir=-1 IFFT
static void fft_inplace(cplx *a, int n, int dir) {
  // bit-reversal
  for (int i=1, j=0; i<n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { cplx tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
  }

  for (int len = 2; len <= n; len <<= 1) {
    float ang = (dir > 0 ? -2.0f : 2.0f) * (float)M_PI / (float)len;
    cplx wlen = { cosf(ang), sinf(ang) };

    for (int i = 0; i < n; i += len) {
      cplx w = {1.0f, 0.0f};
      for (int j = 0; j < len/2; j++) {
        cplx u = a[i+j];
        cplx v = c_mul(a[i+j+len/2], w);
        a[i+j] = c_add(u, v);
        a[i+j+len/2] = c_sub(u, v);
        w = c_mul(w, wlen);
      }
    }
  }

  if (dir < 0) {
    float inv = 1.0f / (float)n;
    for (int i=0;i<n;i++){ a[i].re *= inv; a[i].im *= inv; }
  }
}

// RBJ cookbook biquad
typedef struct {
  float b0,b1,b2,a1,a2;
  float z1,z2; // DF2T states
} Biquad;

static inline float biquad_df2t(Biquad *q, float x) {
  float y = q->b0*x + q->z1;
  q->z1 = q->b1*x - q->a1*y + q->z2;
  q->z2 = q->b2*x - q->a2*y;
  return y;
}

static Biquad biquad_make_hpf(float fs, float f0, float Q) {
  float w0 = 2.0f*(float)M_PI*f0/fs;
  float cw = cosf(w0), sw = sinf(w0);
  float alpha = sw/(2.0f*Q);

  float b0 =  (1.0f + cw)/2.0f;
  float b1 = -(1.0f + cw);
  float b2 =  (1.0f + cw)/2.0f;
  float a0 =  1.0f + alpha;
  float a1 = -2.0f*cw;
  float a2 =  1.0f - alpha;

  Biquad q = {0};
  q.b0=b0/a0; q.b1=b1/a0; q.b2=b2/a0;
  q.a1=a1/a0; q.a2=a2/a0;
  q.z1=0; q.z2=0;
  return q;
}

static Biquad biquad_make_notch(float fs, float f0, float Q) {
  float w0 = 2.0f*(float)M_PI*f0/fs;
  float cw = cosf(w0), sw = sinf(w0);
  float alpha = sw/(2.0f*Q);

  float b0 = 1.0f;
  float b1 = -2.0f*cw;
  float b2 = 1.0f;
  float a0 = 1.0f + alpha;
  float a1 = -2.0f*cw;
  float a2 = 1.0f - alpha;

  Biquad q = {0};
  q.b0=b0/a0; q.b1=b1/a0; q.b2=b2/a0;
  q.a1=a1/a0; q.a2=a2/a0;
  q.z1=0; q.z2=0;
  return q;
}

// DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1]
typedef struct {
  float R;
  float x1;
  float y1;
} DCBlock;

static inline float dc_block(DCBlock *d, float x) {
  float y = x - d->x1 + d->R*d->y1;
  d->x1 = x;
  d->y1 = y;
  return y;
}

// Hann window
static void make_hann(float *w, int N) {
  for (int n=0;n<N;n++) w[n] = 0.5f - 0.5f*cosf(2.0f*(float)M_PI*(float)n/(float)(N-1));
}

// ----------------- Denoiser core (STFT + wiener) -----------------

typedef struct {
  int N;
  int hop;
  int bins; // N/2+1
  float fs;

  // windows
  float *win;

  // spectral state per bin
  float *noise_psd;   // N[k] estimate
  float *xi;          // a-priori SNR
  float *gain_prev;   // prev gain for smoothing

  // time-domain overlap-add
  float *ola;         // length N

  // parameters
  float noise_alpha;      // 0.95..0.995
  float xi_alpha;         // decision-directed 0.9
  float gain_smooth;      // 0.8 typical
  float gmin;             // 0.05..0.15
  float aggr;             // >1.0 more aggressive
  float vad_db;           // threshold dB to consider speech
  int   init_noise_frames; // bootstrap frames from beginning
} Denoiser;

static Denoiser denoiser_create(int N, int hop, float fs) {
  if ((N & (N-1)) != 0) die("N must be power-of-two for this FFT");
  if (hop <= 0 || hop > N) die("Invalid hop");

  Denoiser d;
  memset(&d, 0, sizeof(d));
  d.N = N;
  d.hop = hop;
  d.bins = N/2 + 1;
  d.fs = fs;

  d.win = (float*)malloc((size_t)N*sizeof(float));
  d.noise_psd = (float*)calloc((size_t)d.bins, sizeof(float));
  d.xi = (float*)calloc((size_t)d.bins, sizeof(float));
  d.gain_prev = (float*)calloc((size_t)d.bins, sizeof(float));
  d.ola = (float*)calloc((size_t)N, sizeof(float));
  if (!d.win || !d.noise_psd || !d.xi || !d.gain_prev || !d.ola) die("OOM denoiser");

  make_hann(d.win, N);

  // defaults (bons para voz)
  d.noise_alpha = 0.98f;
  d.xi_alpha = 0.90f;
  d.gain_smooth = 0.80f;
  d.gmin = 0.08f;
  d.aggr = 1.00f;
  d.vad_db = 3.0f;

  d.init_noise_frames = (int)ceilf((0.35f * fs) / (float)hop); // ~350ms bootstrap
  if (d.init_noise_frames < 2) d.init_noise_frames = 2;

  // init gains
  for (int k=0;k<d.bins;k++) d.gain_prev[k] = 1.0f;

  return d;
}

static void denoiser_destroy(Denoiser *d) {
  free(d->win);
  free(d->noise_psd);
  free(d->xi);
  free(d->gain_prev);
  free(d->ola);
  memset(d, 0, sizeof(*d));
}

// VAD simples por SNR de banda (energia espectral vs ruído estimado)
static int vad_is_speech(const float *psd, const float *noise_psd, int bins, float vad_db) {
  double P=0.0, N=0.0;
  // ignora DC e bins muito altos (opcional); aqui usa tudo
  for (int k=1;k<bins;k++) {
    P += psd[k];
    N += noise_psd[k] + 1e-12;
  }
  double snr = P / N;
  double snr_db = 10.0*log10(snr + 1e-12);
  return snr_db > vad_db;
}

// Aplica denoise em um canal (float in/out em [-1,1]), processando por frames.
// out é escrito sequencialmente; preserva comprimento.
static void denoise_channel(
  const float *in, float *out, int n_samples,
  Denoiser *D
) {
  const int N = D->N;
  const int hop = D->hop;
  const int bins = D->bins;

  cplx *X = (cplx*)calloc((size_t)N, sizeof(cplx));
  float *psd = (float*)malloc((size_t)bins*sizeof(float));
  if (!X || !psd) die("OOM FFT buffers");

  int frame_idx = 0;
  int out_pos = 0;

  // zera saída
  for (int i=0;i<n_samples;i++) out[i] = 0.0f;

  for (int start = 0; start < n_samples; start += hop, frame_idx++) {

    // ---- analysis frame: windowed samples ----
    for (int n=0;n<N;n++) {
      int idx = start + n;
      float x = (idx < n_samples) ? in[idx] : 0.0f;
      X[n].re = x * D->win[n];
      X[n].im = 0.0f;
    }

    // FFT
    fft_inplace(X, N, +1);

    // PSD (0..N/2)
    for (int k=0;k<bins;k++) {
      float re = X[k].re, im = X[k].im;
      psd[k] = re*re + im*im;
    }

    // Bootstrap noise: primeiros frames assumidos predominantemente "ruído"
    int speech = 0;
    if (frame_idx < D->init_noise_frames) {
      speech = 0;
    } else {
      speech = vad_is_speech(psd, D->noise_psd, bins, D->vad_db);
    }

    // Atualiza ruído se não fala (ou bootstrap)
    if (!speech) {
      for (int k=0;k<bins;k++) {
        float Nk = D->noise_psd[k];
        float Pk = psd[k];
        D->noise_psd[k] = D->noise_alpha*Nk + (1.0f - D->noise_alpha)*Pk;
      }
    }

    // ---- compute gain & apply ----
    for (int k=0;k<bins;k++) {
      float Nk = D->noise_psd[k] + 1e-12f;
      float gamma = psd[k] / Nk;                 // a-posteriori SNR
      float gamma_m1 = fmaxf(gamma - 1.0f, 0.0f);

      // decisão-dirigida (Ephraim-Malah simplificado)
      // xi = a*xi_prev + (1-a)*max(gamma-1,0)
      float xi_new = D->xi_alpha*D->xi[k] + (1.0f - D->xi_alpha)*gamma_m1;

      // agressividade: "empurra" xi para baixo => mais supressão
      // aggr > 1.0 => mais supressão
      xi_new = xi_new / D->aggr;

      // Wiener gain
      float G = xi_new / (1.0f + xi_new);

      // gain floor
      if (G < D->gmin) G = D->gmin;
      if (G > 1.0f) G = 1.0f;

      // smoothing temporal do ganho
      G = D->gain_smooth*D->gain_prev[k] + (1.0f - D->gain_smooth)*G;

      D->xi[k] = xi_new;
      D->gain_prev[k] = G;

      // aplica em bin positivo
      X[k].re *= G;
      X[k].im *= G;

      // aplica simetria (bins negativos) para manter sinal real
      if (k > 0 && k < bins-1) {
        int km = N - k;
        X[km].re *= G;
        X[km].im *= G;
      }
    }

    // IFFT
    fft_inplace(X, N, -1);

    // ---- synthesis + overlap-add ----
    // Usa janela Hann também na síntese; com hop 50% Hann é perfeito (COLA).
    // Em 75%, ainda funciona bem; aqui mantemos Hann e confiança prática.
    // OLA buffer D->ola acumula contribuições.
    for (int n=0;n<N;n++) {
      float y = X[n].re * D->win[n];
      D->ola[n] += y;
    }

    // emite hop samples
    for (int n=0;n<hop;n++) {
      int idx = start + n;
      if (idx < n_samples) out[idx] = D->ola[n];
    }

    // shift OLA buffer: descarta primeiros hop e move o resto
    memmove(D->ola, D->ola + hop, (size_t)(N - hop)*sizeof(float));
    memset(D->ola + (N - hop), 0, (size_t)hop*sizeof(float));

    out_pos = start + hop;
    (void)out_pos;
  }

  free(psd);
  free(X);
}

// ----------------- CLI / Main -----------------

static void usage(const char *argv0) {
  fprintf(stderr,
    "Usage:\n"
    "  %s <in.wav> <out.wav> [options]\n\n"
    "Options:\n"
    "  --N <1024>       FFT size (power of 2)\n"
    "  --hop <256>      hop size\n"
    "  --hp <100>       high-pass cutoff Hz (0 disables)\n"
    "  --notch <0|50|60> notch freq Hz (0 disables)\n"
    "  --gmin <0.08>    gain floor (0.05..0.15)\n"
    "  --aggr <1.0>     aggressiveness (>1 = more suppression)\n"
    "  --vad_db <3.0>   VAD threshold in dB\n",
    argv0
  );
}

static int arg_eq(const char *a, const char *b){ return strcmp(a,b)==0; }

int main(int argc, char **argv) {
  if (argc < 3) { usage(argv[0]); return 1; }

  const char *in_path = argv[1];
  const char *out_path = argv[2];

  int N = 1024;
  int hop = 256;
  float hp_hz = 100.0f;
  float notch_hz = 0.0f;
  float gmin = 0.08f;
  float aggr = 1.00f;
  float vad_db = 3.0f;

  for (int i=3; i<argc; i++) {
    if (arg_eq(argv[i], "--N") && i+1<argc) { N = atoi(argv[++i]); continue; }
    if (arg_eq(argv[i], "--hop") && i+1<argc) { hop = atoi(argv[++i]); continue; }
    if (arg_eq(argv[i], "--hp") && i+1<argc) { hp_hz = (float)atof(argv[++i]); continue; }
    if (arg_eq(argv[i], "--notch") && i+1<argc) { notch_hz = (float)atof(argv[++i]); continue; }
    if (arg_eq(argv[i], "--gmin") && i+1<argc) { gmin = (float)atof(argv[++i]); continue; }
    if (arg_eq(argv[i], "--aggr") && i+1<argc) { aggr = (float)atof(argv[++i]); continue; }
    if (arg_eq(argv[i], "--vad_db") && i+1<argc) { vad_db = (float)atof(argv[++i]); continue; }
    usage(argv[0]); die("Unknown/invalid option");
  }

  FILE *in = fopen(in_path, "rb");
  if (!in) die_errno("fopen(input)");
  WavInfo info = parse_wav(in);

  if (info.fmt.num_channels != 1 && info.fmt.num_channels != 2) {
    die("Only mono/stereo supported");
  }

  if (fseek(in, info.data_offset, SEEK_SET) != 0) die_errno("fseek(data_offset)");

  int channels = (int)info.fmt.num_channels;
  int frame_size = channels * 2;
  if (info.data_size % (uint32_t)frame_size != 0) die("data_size not multiple of frame_size");

  int frames = (int)(info.data_size / (uint32_t)frame_size);
  size_t n_samples_total = (size_t)frames * (size_t)channels;

  int16_t *pcm_in = (int16_t*)malloc(n_samples_total * sizeof(int16_t));
  int16_t *pcm_out = (int16_t*)malloc(n_samples_total * sizeof(int16_t));
  if (!pcm_in || !pcm_out) die("OOM pcm buffers");

  if (fread(pcm_in, sizeof(int16_t), n_samples_total, in) != n_samples_total) {
    die("Failed to read samples");
  }
  fclose(in);

  float fs = (float)info.fmt.sample_rate;

  // Pré-filtros por canal
  DCBlock *dc = (DCBlock*)calloc((size_t)channels, sizeof(DCBlock));
  Biquad *hpf = (Biquad*)calloc((size_t)channels, sizeof(Biquad));
  Biquad *notch = (Biquad*)calloc((size_t)channels, sizeof(Biquad));
  if (!dc || !hpf || !notch) die("OOM filters");

  for (int ch=0; ch<channels; ch++) {
    dc[ch].R = 0.995f;
    if (hp_hz > 0.0f) hpf[ch] = biquad_make_hpf(fs, hp_hz, 0.707f);
    if (notch_hz > 0.0f) notch[ch] = biquad_make_notch(fs, notch_hz, 10.0f);
  }

  // Converte para float por canal e aplica pré-filtros
  float **xch = (float**)calloc((size_t)channels, sizeof(float*));
  float **ych = (float**)calloc((size_t)channels, sizeof(float*));
  if (!xch || !ych) die("OOM channel arrays");

  for (int ch=0; ch<channels; ch++) {
    xch[ch] = (float*)malloc((size_t)frames * sizeof(float));
    ych[ch] = (float*)malloc((size_t)frames * sizeof(float));
    if (!xch[ch] || !ych[ch]) die("OOM channel buffers");
  }

  for (int i=0; i<frames; i++) {
    for (int ch=0; ch<channels; ch++) {
      int16_t s = pcm_in[i*channels + ch];
      float x = (float)s / 32768.0f;

      x = dc_block(&dc[ch], x);
      if (hp_hz > 0.0f) x = biquad_df2t(&hpf[ch], x);
      if (notch_hz > 0.0f) x = biquad_df2t(&notch[ch], x);

      xch[ch][i] = x;
    }
  }

  // Denoiser por canal (se stereo, processa cada canal independentemente)
  for (int ch=0; ch<channels; ch++) {
    Denoiser D = denoiser_create(N, hop, fs);
    D.gmin = gmin;
    D.aggr = aggr;
    D.vad_db = vad_db;

    denoise_channel(xch[ch], ych[ch], frames, &D);
    denoiser_destroy(&D);
  }

  // Float -> PCM16 (com saturação)
  for (int i=0; i<frames; i++) {
    for (int ch=0; ch<channels; ch++) {
      float y = ych[ch][i];
      // limiter simples
      y = clampf(y, -1.0f, 1.0f);
      int32_t si = (int32_t)lrintf(y * 32767.0f);
      pcm_out[i*channels + ch] = sat16i(si);
    }
  }

  // Write output
  FILE *out = fopen(out_path, "wb");
  if (!out) die_errno("fopen(output)");

  WavFmt outfmt = info.fmt;
  outfmt.block_align = (uint16_t)(outfmt.num_channels * 2);
  outfmt.byte_rate = outfmt.sample_rate * (uint32_t)outfmt.block_align;

  write_wav_header(out, &outfmt, info.data_size);
  if (fwrite(pcm_out, sizeof(int16_t), n_samples_total, out) != n_samples_total) die("Failed to write samples");
  fclose(out);

  // cleanup
  for (int ch=0; ch<channels; ch++) { free(xch[ch]); free(ych[ch]); }
  free(xch); free(ych);
  free(dc); free(hpf); free(notch);
  free(pcm_in); free(pcm_out);

  return 0;
}
