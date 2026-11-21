// The two hot loops of an HSDM node, for WebAssembly: keyed BLAKE2b (how a feature becomes
// signed positions in a hypervector) and the AND + popcount scan over packed ternary vectors.
// Everything else is in hsdm.mjs. Build: see build-wasm.sh. No libc.

typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

#define EXPORT(name) __attribute__((export_name(name)))

static const u64 IV[8] = {
  0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
  0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL, 0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
};

static const u8 SIGMA[12][16] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
  {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
  {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
  {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
  {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
  {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
  {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
  {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
};

static inline u64 rotr(u64 x, int n) { return (x >> n) | (x << (64 - n)); }

static inline u64 load64(const u8 *p) {
  u64 x = 0;
  for (int i = 7; i >= 0; i--) x = (x << 8) | p[i];
  return x;
}

#define G(a, b, c, d, x, y)                     \
  v[a] = v[a] + v[b] + (x); v[d] = rotr(v[d] ^ v[a], 32); \
  v[c] = v[c] + v[d];       v[b] = rotr(v[b] ^ v[c], 24); \
  v[a] = v[a] + v[b] + (y); v[d] = rotr(v[d] ^ v[a], 16); \
  v[c] = v[c] + v[d];       v[b] = rotr(v[b] ^ v[c], 63);

static void compress(u64 h[8], const u8 block[128], u64 t, int last) {
  u64 v[16], m[16];
  for (int i = 0; i < 16; i++) m[i] = load64(block + 8 * i);
  for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = IV[i]; }
  v[12] ^= t;                       // message length so far; never above 2^64 here
  if (last) v[14] = ~v[14];
  for (int r = 0; r < 12; r++) {
    const u8 *s = SIGMA[r];
    G(0, 4, 8, 12, m[s[0]], m[s[1]]);
    G(1, 5, 9, 13, m[s[2]], m[s[3]]);
    G(2, 6, 10, 14, m[s[4]], m[s[5]]);
    G(3, 7, 11, 15, m[s[6]], m[s[7]]);
    G(0, 5, 10, 15, m[s[8]], m[s[9]]);
    G(1, 6, 11, 12, m[s[10]], m[s[11]]);
    G(2, 7, 8, 13, m[s[12]], m[s[13]]);
    G(3, 4, 9, 14, m[s[14]], m[s[15]]);
  }
  for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

// State after the key block, for each of the two keyed hashes HSDM uses:
//   features: blake2b(feature, key=K, digest_size=64)
//   vocab:    blake2b(word, key=K, digest_size=8, person=b"vocab")
static u64 feature_state[8], vocab_state[8];
static u8 block[128];

static void keyed_init(u64 h[8], const u8 key[32], u32 outlen, const u8 *person) {
  for (int i = 0; i < 8; i++) h[i] = IV[i];
  h[0] ^= 0x01010000ULL ^ ((u64)32 << 8) ^ outlen;
  if (person) { h[6] ^= load64(person); h[7] ^= load64(person + 8); }
  for (int i = 0; i < 128; i++) block[i] = i < 32 ? key[i] : 0;
  compress(h, block, 128, 0);       // the key block; HSDM never hashes an empty message
}

static void keyed_hash(const u64 start[8], const u8 *msg, u32 len, u64 h[8]) {
  for (int i = 0; i < 8; i++) h[i] = start[i];
  u64 t = 128;
  while (len > 128) {
    t += 128;
    compress(h, msg, t, 0);
    msg += 128;
    len -= 128;
  }
  for (u32 i = 0; i < 128; i++) block[i] = i < len ? msg[i] : 0;
  compress(h, block, t + len, 1);
}

// Plain (unkeyed) BLAKE2b, for deriving the encoder key from the swarm key.
EXPORT("hash") void hash(const u8 *msg, u32 len, u32 outlen, u64 *out) {
  for (int i = 0; i < 8; i++) out[i] = IV[i];
  out[0] ^= 0x01010000ULL ^ outlen;
  u64 t = 0;
  while (len > 128) {
    t += 128;
    compress(out, msg, t, 0);
    msg += 128;
    len -= 128;
  }
  for (u32 i = 0; i < 128; i++) block[i] = i < len ? msg[i] : 0;
  compress(out, block, t + len, 1);
}

EXPORT("init") void init(const u8 *key) {
  static const u8 person[16] = {'v', 'o', 'c', 'a', 'b'};
  keyed_init(feature_state, key, 64, 0);
  keyed_init(vocab_state, key, 8, person);
}

// 64 bytes = the 32 little-endian uint16 positions and signs of one feature
EXPORT("hash_feature") void hash_feature(const u8 *msg, u32 len, u64 *out) {
  keyed_hash(feature_state, msg, len, out);
}

// The low 32 bits of the 8-byte vocab digest, little-endian (the sketch takes it mod 16384)
EXPORT("hash_vocab") u32 hash_vocab(const u8 *msg, u32 len) {
  u64 h[8];
  keyed_hash(vocab_state, msg, len, h);
  return (u32)h[0];
}

// Ternary vectors are two bit masks, 32 words each at dim 2048. q holds the query's
// positive words then its negative words. out[i] = (agree - clash) for row i: a trit can't be
// both + and -, so the two ANDs in each count never overlap and one popcount of their OR does.
EXPORT("scan") void scan(const u64 *pos, const u64 *neg, u32 rows, u32 words, const u64 *q, int *out) {
  const u64 *qp = q, *qn = q + words;
  for (u32 i = 0; i < rows; i++) {
    const u64 *p = pos + (u64)i * words, *n = neg + (u64)i * words;
    int agree = 0, clash = 0;
    for (u32 j = 0; j < words; j++) {
      agree += __builtin_popcountll((p[j] & qp[j]) | (n[j] & qn[j]));
      clash += __builtin_popcountll((p[j] & qn[j]) | (n[j] & qp[j]));
    }
    out[i] = agree - clash;
  }
}

// The same for chosen rows only: what a walk across a few cells reads.
EXPORT("scan_rows") void scan_rows(const u64 *pos, const u64 *neg, const u32 *rows, u32 count, u32 words,
                                   const u64 *q, int *out) {
  const u64 *qp = q, *qn = q + words;
  for (u32 k = 0; k < count; k++) {
    const u64 *p = pos + (u64)rows[k] * words, *n = neg + (u64)rows[k] * words;
    int agree = 0, clash = 0;
    for (u32 j = 0; j < words; j++) {
      agree += __builtin_popcountll((p[j] & qp[j]) | (n[j] & qn[j]));
      clash += __builtin_popcountll((p[j] & qn[j]) | (n[j] & qp[j]));
    }
    out[k] = agree - clash;
  }
}

// Non-zero trits per row.
EXPORT("nnz") void nnz(const u64 *pos, const u64 *neg, u32 rows, u32 words, int *out) {
  for (u32 i = 0; i < rows; i++) {
    int c = 0;
    for (u32 j = 0; j < words; j++) c += __builtin_popcountll(pos[(u64)i * words + j]) + __builtin_popcountll(neg[(u64)i * words + j]);
    out[i] = c;
  }
}
