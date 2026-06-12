/*
 * yespower_smix.cl - Clean GPU smix kernel (YESPOWER_1.0)
 */
typedef unsigned int u32;
typedef unsigned long u64;

#define N_CONST     2048u
#define R_CONST     32u
#define SL          (32u * R_CONST)
#define PWXsimple   2
#define PWXgather   4
#define PWXrounds   3
#define PWXbytes    (PWXgather * PWXsimple * 8)
#define PWXwords    (PWXbytes / 4)
#define SWidth      11
#define Swidth_to_Sbytes1  ((1u << SWidth) * PWXsimple * 8)
#define Swidth_to_Smask    (((1u << SWidth) - 1u) * PWXsimple * 8)
#define Sbytes       (3u * Swidth_to_Sbytes1)
#define Sbytes_words (Sbytes / 4)
#define Ssegment     (Sbytes_words / 3)

#define ROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))

typedef struct { u32 so, s1, s2, w; } sctx;

/* ==================== Salsa20/2 core ==================== */
static void salsa20_2(u32 B[16]) {
    u32 x[16];
    for (u32 i = 0; i < 16; i++) x[i * 5 % 16] = B[i];
    x[4] ^= ROTL(x[0]+x[12],7);  x[8] ^= ROTL(x[4]+x[0],9);
    x[12] ^= ROTL(x[8]+x[4],13); x[0] ^= ROTL(x[12]+x[8],18);
    x[9] ^= ROTL(x[5]+x[1],7);   x[13] ^= ROTL(x[9]+x[5],9);
    x[1] ^= ROTL(x[13]+x[9],13); x[5] ^= ROTL(x[1]+x[13],18);
    x[14] ^= ROTL(x[10]+x[6],7); x[2] ^= ROTL(x[14]+x[10],9);
    x[6] ^= ROTL(x[2]+x[14],13); x[10] ^= ROTL(x[6]+x[2],18);
    x[3] ^= ROTL(x[15]+x[11],7); x[7] ^= ROTL(x[3]+x[15],9);
    x[11] ^= ROTL(x[7]+x[3],13); x[15] ^= ROTL(x[11]+x[7],18);
    x[1] ^= ROTL(x[0]+x[3],7);   x[2] ^= ROTL(x[1]+x[0],9);
    x[3] ^= ROTL(x[2]+x[1],13);  x[0] ^= ROTL(x[3]+x[2],18);
    x[6] ^= ROTL(x[5]+x[4],7);   x[7] ^= ROTL(x[6]+x[5],9);
    x[4] ^= ROTL(x[7]+x[6],13);  x[5] ^= ROTL(x[4]+x[7],18);
    x[11] ^= ROTL(x[10]+x[9],7); x[8] ^= ROTL(x[11]+x[10],9);
    x[9] ^= ROTL(x[8]+x[11],13); x[10] ^= ROTL(x[9]+x[8],18);
    x[12] ^= ROTL(x[15]+x[14],7); x[13] ^= ROTL(x[12]+x[15],9);
    x[14] ^= ROTL(x[13]+x[12],13); x[15] ^= ROTL(x[14]+x[13],18);
    for (u32 i = 0; i < 16; i++) B[i] += x[i * 5 % 16];
}

/* ==================== pwxform with S evolution ==================== */
static void pwxform(u32 X[PWXwords], __global u32 *S_base, sctx *sc) {
    __global u32 (*S0)[2] = (__global u32 (*)[2])(S_base + sc->so);
    __global u32 (*S1)[2] = (__global u32 (*)[2])(S_base + sc->s1);
    u32 w = sc->w;
    u32 Sm = Swidth_to_Smask;
    for (u32 i = 0; i < PWXrounds; i++) {
        for (u32 j = 0; j < PWXgather; j++) {
            u32 *Xj = X + j * 4;
            u32 xl = Xj[0], xh = Xj[1];
            __global u32 (*p0)[2] = S0 + ((xl & Sm) / 8u);
            __global u32 (*p1)[2] = S1 + ((xh & Sm) / 8u);
            for (u32 k = 0; k < PWXsimple; k++) {
                u64 s0v = ((u64)p0[k][1] << 32) + (u64)p0[k][0];
                u64 s1v = ((u64)p1[k][1] << 32) + (u64)p1[k][0];
                xl = Xj[k * 2]; xh = Xj[k * 2 + 1];
                u64 xv = (u64)xh * (u64)xl;
                xv += s0v; xv ^= s1v;
                Xj[k * 2] = (u32)xv; Xj[k * 2 + 1] = (u32)(xv >> 32);
            }
            if ((i == 0) || (j < PWXgather / 2)) {
                if (j & 1) {
                    for (u32 k = 0; k < PWXsimple; k++) {
                        S1[w][0] = Xj[k * 2]; S1[w][1] = Xj[k * 2 + 1]; w++;
                    }
                } else {
                    for (u32 k = 0; k < PWXsimple; k++) {
                        S0[w + k][0] = Xj[k * 2]; S0[w + k][1] = Xj[k * 2 + 1];
                    }
                }
            }
        }
    }
    /* Rotate S pointers: (S0,S1,S2) <- (S2,S0,S1) */
    u32 old_so = sc->so; sc->so = sc->s2; sc->s2 = sc->s1; sc->s1 = old_so;
    sc->w = w & ((1u << SWidth) * PWXsimple - 1u);
}

/* ==================== blockmix_pwxform helpers ==================== */
static void blockmix_pwxform(u32 *B, u32 r, __global u32 *S_base, sctx *sc) {
    u32 r1 = 128u * r / PWXbytes;
    u32 Xt[PWXwords];
    for (u32 i = 0; i < PWXwords; i++) Xt[i] = B[(r1 - 1) * PWXwords + i];
    for (u32 i0 = 0; i0 < r1; i0++) {
        if (r1 > 1) { for (u32 k = 0; k < PWXwords; k++) Xt[k] ^= B[i0 * PWXwords + k]; }
        pwxform(Xt, S_base, sc);
        for (u32 k = 0; k < PWXwords; k++) B[i0 * PWXwords + k] = Xt[k];
    }
    u32 salsa_idx = (r1 - 1) * PWXbytes / 64u;
    salsa20_2(&B[salsa_idx * 16]);
}

/* ==================== blockmix_salsa for S-init (r=1) ==================== */
static void blockmix_salsa_r1(u32 *B) {
    u32 Xt[16];
    for (u32 i = 0; i < 16; i++) Xt[i] = B[16 + i];
    for (u32 i0 = 0; i0 < 2; i0++) {
        for (u32 k = 0; k < 16; k++) Xt[k] ^= B[i0 * 16 + k];
        salsa20_2(Xt);
        for (u32 k = 0; k < 16; k++) B[i0 * 16 + k] = Xt[k];
    }
}

/* S-box initialization: smix1 with r=1, V=S, blockmix_salsa */
static void sinit(u32 *X, __global u32 *S_base, u32 Nloop) {
    for (u32 i = 0; i < Nloop; i++) {
        for (u32 k = 0; k < 32; k++) S_base[i * 32 + k] = X[k];
        if (i > 1) {
            u32 intfy = X[16];
            u32 n = i; while (n & (n - 1)) n &= (n - 1);
            u32 j = (intfy & (n - 1)) + (i - n);
            for (u32 k = 0; k < 32; k++) X[k] ^= S_base[j * 32 + k];
        }
        blockmix_salsa_r1(X);
    }
}

/* ==================== smix functions ==================== */
static void smix(u32 *X, u32 *B, __global u32 *V, __global u32 *S_base, sctx *sc,
                 u32 r, u32 N, u32 V_off, u32 Nloop_rw) {
    /* 1: X <-- B (SIMD unshuffle) */
    for (u32 b = 0; b < 64; b++)
        for (u32 i = 0; i < 16; i++)
            X[b * 16 + i] = B[b * 16 + (i * 5 % 16)];

    /* Precompute k=1..r-1 (YESPOWER_1.0) */
    for (u32 k = 1; k < r; k++) {
        for (u32 i = 0; i < 32; i++) X[k * 32 + i] = X[(k - 1) * 32 + i];
        blockmix_pwxform(&X[k * 32], 1, S_base, sc);
    }

    /* smix1: for i = 0 to N-1 */
    for (u32 i = 0; i < N; i++) {
        for (u32 k = 0; k < SL; k++) V[V_off + i * SL + k] = X[k];
        if (i > 1) {
            u32 intfy = X[(2 * r - 1) * 16];
            u32 n = i; while (n & (n - 1)) n &= (n - 1);
            u32 j = (intfy & (n - 1)) + (i - n);
            for (u32 k = 0; k < SL; k++) X[k] ^= V[V_off + j * SL + k];
        }
        blockmix_pwxform(X, r, S_base, sc);
    }

    /* B' <-- X (SIMD shuffle) */
    for (u32 b = 0; b < 64; b++)
        for (u32 i = 0; i < 16; i++)
            B[b * 16 + (i * 5 % 16)] = X[b * 16 + i];

    /* smix2: X <-- B */
    for (u32 b = 0; b < 64; b++)
        for (u32 i = 0; i < 16; i++)
            X[b * 16 + i] = B[b * 16 + (i * 5 % 16)];

    Nloop_rw = (N + 2) / 3; Nloop_rw++; Nloop_rw &= ~1u;
    for (u32 i = 0; i < Nloop_rw; i++) {
        u32 j = X[(2 * r - 1) * 16] & (N - 1);
        for (u32 k = 0; k < SL; k++) X[k] ^= V[V_off + j * SL + k];
        for (u32 k = 0; k < SL; k++) V[V_off + j * SL + k] = X[k];
        blockmix_pwxform(X, r, S_base, sc);
    }

    for (u32 b = 0; b < 64; b++)
        for (u32 i = 0; i < 16; i++)
            B[b * 16 + (i * 5 % 16)] = X[b * 16 + i];
}

__kernel void yespower_smix(
    __global u32 *Bi_g,
    __global u32 *V_g,
    __global u32 *S_g,
    u32 start_wg,
    u32 num_wgs
) {
    u32 gid = get_global_id(0);
    if (gid >= num_wgs) return;

    u32 V_off = gid * N_CONST * SL;
    __global u32 *S_base = S_g + gid * 3 * Ssegment;

    sctx sc;
    sc.so = 0; sc.s1 = Ssegment; sc.s2 = 2 * Ssegment; sc.w = 0;

    /* Local arrays */
    u32 X[1024], B[1024], Xs[32];
    for (u32 k = 0; k < 1024; k++) B[k] = Bi_g[gid * 1024 + k];

    /* === S-box initialization === */
    /* Copy Bi[0..127] to Xs with SIMD unshuffle */
    for (u32 k = 0; k < 2; k++)
        for (u32 i = 0; i < 16; i++)
            Xs[k * 16 + i] = B[k * 16 + (i * 5 % 16)];
    
    u32 N_sinit = Sbytes / 128u;  /* 98304 / 128 = 768 */
    sinit(Xs, S_base, N_sinit);
    
    /* Write Xs back to B[0..127] with SIMD shuffle */
    for (u32 k = 0; k < 2; k++)
        for (u32 i = 0; i < 16; i++)
            B[k * 16 + (i * 5 % 16)] = Xs[k * 16 + i];
    
    /* Reset S-box context after S-init (sinit doesn't use pwxform) */
    sc.so = 0; sc.s1 = Ssegment; sc.s2 = 2 * Ssegment; sc.w = 0;

    /* === Main smix (uses pre-initialized S) === */
    smix(X, B, V_g, S_base, &sc, R_CONST, N_CONST, V_off, 0);

    for (u32 k = 0; k < 1024; k++) Bi_g[gid * 1024 + k] = B[k];
}
