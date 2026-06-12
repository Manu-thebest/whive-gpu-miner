/*
 * test_hybrid_full.c - Full hybrid test: CPU PBKDF2 + GPU smix + CPU HMAC
 *
 * Build:
 *   gcc -O2 -o test_hybrid_full test_hybrid_full.c \
 *       -Iwhive-cpuminer-ref/yespower \
 *       -lOpenCL -lm \
 *       whive-cpuminer-ref/yespower/yespower-ref.o \
 *       whive-cpuminer-ref/yespower/sha256.o
 *
 * Run: ./test_hybrid_full
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <CL/cl.h>

#include <errno.h>
#include "sha256.h"
#include "yespower.h"
#include "sysendian.h"

/* Internal functions from yespower-ref.c that we need */
/* We'll extract just what we need */

#define CHECK(msg, err) do { \
    if (err != CL_SUCCESS) { \
        printf("FAIL: %s -> %d\n", msg, err); exit(1); \
    } } while(0)

/* Constants matching yespower-ref.c */
#define PWXsimple 2
#define PWXgather 4
#define PWXrounds_1_0 3
#define Swidth_1_0 11
#define PWXbytes (PWXgather * PWXsimple * 8)
#define PWXwords (PWXbytes / sizeof(uint32_t))
#define Swidth_to_Sbytes1(Swidth) ((1 << Swidth) * PWXsimple * 8)
#define Swidth_to_Smask(Swidth) (((1 << Swidth) - 1) * PWXsimple * 8)
#define Sbytes (3 * Swidth_to_Sbytes1(Swidth_1_0))  /* 98304 */
#define Smask Swidth_to_Smask(Swidth_1_0)             /* 32752 */
#define SB1 (Swidth_to_Sbytes1(Swidth_1_0))           /* 32768 */
#define S0W (SB1 / 4)                                 /* 8192 words */

static void print_hash(const char *label, const uint8_t *data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len; i++) printf("%02x", data[i]);
    printf("\n");
}

static void write_binary(const char *path, const uint8_t *data, size_t len) {
    FILE *fp = fopen(path, "wb");
    if (fp) {
        fwrite(data, 1, len, fp);
        fclose(fp);
    }
}

static void print_words(const char *label, const uint32_t *data, int n) {
    printf("%s:", label);
    for (int i = 0; i < n && i < 16; i++) printf(" %08x", data[i]);
    if (n > 16) printf(" ...");
    printf("\n");
}

/* ===== CPU reference functions ===== */

/* Run the full yespower_tls to get reference result */
static void do_cpu_reference(yespower_binary_t *result) {
    yespower_params_t yp;
    yp.version = YESPOWER_1_0;
    yp.N = 2048;
    yp.r = 32;
    yp.pers = NULL;
    yp.perslen = 0;

    uint8_t header[80];
    memset(header, 0, 80);
    header[0] = 0x20;
    header[76] = 0x14; header[77] = 0xF9;
    header[78] = 0x2B; header[79] = 0x6A;

    int rc = yespower_tls(header, 80, &yp, result);
    printf("CPU reference: rc=%d\n", rc);
    print_hash("CPU final hash", result->uc, 32);
}

/* Compute step 1: SHA256(header) -> sha256_out */
static void step1_sha256(const uint8_t *header, uint8_t *sha256_out) {
    SHA256_Buf(header, 80, sha256_out);
    print_hash("Step1: SHA256(header)", sha256_out, 32);
}

/* Compute step 2: PBKDF2(sha256, empty_salt, 1, Bi, 4096) */
/* Note: For YESPOWER_1.0 with pers=NULL, the PBKDF2 salt is empty */
static void step2_pbkdf2(const uint8_t *sha256, uint8_t *Bi) {
    PBKDF2_SHA256(sha256, 32, NULL, 0, 1, Bi, 4096);
    print_hash("Step2: PBKDF2 Bi[0..7]", Bi, 8);
    print_hash("Step2: PBKDF2 Bi[4032..]", Bi + 4032, 32);
}

/* Save the key (first 32 bytes of Bi before smix) */
static void save_key(const uint8_t *Bi, uint8_t *key) {
    memcpy(key, Bi, 32);
    print_hash("Key (first 32B of Bi)", key, 32);
}

/* ===== S-box initialization (copied from reference) ===== */

/* Salsa20 core */
static void salsa20(uint32_t B[16], uint32_t rounds) {
    uint32_t x[16];
    size_t i;
    for (i = 0; i < 16; i++)
        x[i * 5 % 16] = B[i];
    for (i = 0; i < rounds; i += 2) {
#define R(a,b) (((a) << (b)) | ((a) >> (32 - (b))))
        x[ 4] ^= R(x[ 0]+x[12], 7);  x[ 8] ^= R(x[ 4]+x[ 0], 9);
        x[12] ^= R(x[ 8]+x[ 4],13);  x[ 0] ^= R(x[12]+x[ 8],18);
        x[ 9] ^= R(x[ 5]+x[ 1], 7);  x[13] ^= R(x[ 9]+x[ 5], 9);
        x[ 1] ^= R(x[13]+x[ 9],13);  x[ 5] ^= R(x[ 1]+x[13],18);
        x[14] ^= R(x[10]+x[ 6], 7);  x[ 2] ^= R(x[14]+x[10], 9);
        x[ 6] ^= R(x[ 2]+x[14],13);  x[10] ^= R(x[ 6]+x[ 2],18);
        x[ 3] ^= R(x[15]+x[11], 7);  x[ 7] ^= R(x[ 3]+x[15], 9);
        x[11] ^= R(x[ 7]+x[ 3],13);  x[15] ^= R(x[11]+x[ 7],18);
        x[ 1] ^= R(x[ 0]+x[ 3], 7);  x[ 2] ^= R(x[ 1]+x[ 0], 9);
        x[ 3] ^= R(x[ 2]+x[ 1],13);  x[ 0] ^= R(x[ 3]+x[ 2],18);
        x[ 6] ^= R(x[ 5]+x[ 4], 7);  x[ 7] ^= R(x[ 6]+x[ 5], 9);
        x[ 4] ^= R(x[ 7]+x[ 6],13);  x[ 5] ^= R(x[ 4]+x[ 7],18);
        x[11] ^= R(x[10]+x[ 9], 7);  x[ 8] ^= R(x[11]+x[10], 9);
        x[ 9] ^= R(x[ 8]+x[11],13);  x[10] ^= R(x[ 9]+x[ 8],18);
        x[12] ^= R(x[15]+x[14], 7);  x[13] ^= R(x[12]+x[15], 9);
        x[14] ^= R(x[13]+x[12],13);  x[15] ^= R(x[14]+x[13],18);
#undef R
    }
    for (i = 0; i < 16; i++)
        B[i] += x[i * 5 % 16];
}

static void blockmix_salsa(uint32_t *B, uint32_t rounds) {
    uint32_t X[16];
    size_t i;
    for (i = 0; i < 16; i++) X[i] = B[16 + i];
    for (i = 0; i < 2; i++) {
        for (size_t k = 0; k < 16; k++) X[k] ^= B[i * 16 + k];
        salsa20(X, rounds);
        for (size_t k = 0; k < 16; k++) B[i * 16 + k] = X[k];
    }
}

/* integerify - same as reference */
static uint32_t integerify(const uint32_t *B, size_t r) {
    const uint32_t *X = &B[(2 * r - 1) * 16];
    return X[0];
}

/* p2floor - largest power of 2 not greater than argument */
static uint32_t p2floor(uint32_t x) {
    uint32_t y;
    while ((y = x & (x - 1))) x = y;
    return x;
}

/* wrap - wrap x to range 0..i-1 */
static uint32_t wrap(uint32_t x, uint32_t i) {
    uint32_t n = p2floor(i);
    return (x & (n - 1)) + (i - n);
}

/* Initialize S-boxes: same as smix1(B, 1, Sbytes/128, S, X, ctx) with V=S */
/* NOTE: This modifies Bi (writes shuffled X back), matching reference behavior */
static void init_S_boxes(uint32_t *Bi, uint32_t *S, uint32_t salsa_rounds) {
    size_t s = 32; /* 32 * r with r=1 */
    uint32_t Nloop = Sbytes / 128; /* 98304 / 128 = 768 */
    uint32_t X[32];

    printf("Initializing S-boxes: Nloop=%u, Sbytes=%u\n", Nloop, Sbytes);

    /* X <-- B (SIMD unshuffle + LE decode) */
    for (size_t k = 0; k < 2; k++)
        for (size_t i = 0; i < 16; i++)
            X[k * 16 + i] = le32dec((uint8_t *)&Bi[k * 16 + (i * 5 % 16)]);

    for (uint32_t i = 0; i < Nloop; i++) {
        /* V_i <-- X (s words) */
        for (size_t k = 0; k < s; k++)
            S[i * s + k] = X[k];
        if (i > 1) {
            uint32_t j = wrap(integerify(X, 1), i);
            for (size_t k = 0; k < s; k++)
                X[k] ^= S[j * s + k];
        }
        /* blockmix_salsa(X, rounds) */
        blockmix_salsa(X, salsa_rounds);
    }

    /* Write X back to B (SIMD shuffle + LE encode), matching smix1 exit */
    for (size_t k = 0; k < 2; k++)
        for (size_t i = 0; i < 16; i++)
            le32enc((uint8_t *)&Bi[k * 16 + (i * 5 % 16)], X[k * 16 + i]);

    printf("S-boxes initialized (%u bytes), B modified\n", Sbytes);
}

/* ===== GPU OpenCL setup ===== */

static cl_context create_context(cl_device_id *device) {
    cl_platform_id platform;
    cl_int err;

    /* Get first platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK("clGetPlatformIDs", err);

    /* Get a GPU device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, device, NULL);
    CHECK("clGetDeviceIDs", err);

    /* Print device info */
    char name[128];
    clGetDeviceInfo(*device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    printf("GPU: %s\n", name);

    cl_context context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
    CHECK("clCreateContext", err);
    return context;
}

static cl_command_queue create_queue(cl_context context, cl_device_id device) {
    cl_int err;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK("clCreateCommandQueue", err);
    return queue;
}

static cl_program build_program(cl_context context, cl_device_id device, const char *source) {
    cl_int err;
    const char *sources[] = { source };
    cl_program program = clCreateProgramWithSource(context, 1, sources, NULL, &err);
    CHECK("clCreateProgramWithSource", err);

    err = clBuildProgram(program, 1, &device, "-cl-std=CL3.0 -cl-opt-disable", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build FAIL:\n%s\n", log);
        exit(1);
    }
    printf("Kernel built OK\n");
    return program;
}

/* ===== Main test ===== */

int main() {
    printf("=== FULL HYBRID TEST ===\n\n");

    /* 1. CPU reference */
    yespower_binary_t reference;
    do_cpu_reference(&reference);

    /* 2. Compute header -> sha256 -> PBKDF2 -> Bi */
    uint8_t header[80];
    memset(header, 0, 80);
    header[0] = 0x20;
    header[76] = 0x14; header[77] = 0xF9;
    header[78] = 0x2B; header[79] = 0x6A;

    uint8_t Bi[4096];
    uint8_t key[32];
    uint8_t sha256_out[32];

    step1_sha256(header, sha256_out);
    step2_pbkdf2(sha256_out, Bi);
    save_key(Bi, key);

    /* S-box init is now done ON GPU. We just upload raw Bi. */
    printf("Skipping CPU S-init (GPU handles it)\n");

    /* 4. OpenCL setup */
    cl_device_id device;
    cl_context context = create_context(&device);
    cl_command_queue queue = create_queue(context, device);

    /* Read kernel source */
    const char *kernel_path = "yespower_smix.cl";
    FILE *fp = fopen(kernel_path, "rb");
    if (!fp) { printf("Can't open %s\n", kernel_path); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    char *kernel_src = malloc(fsize + 1);
    fread(kernel_src, 1, fsize, fp);
    kernel_src[fsize] = 0;
    fclose(fp);

    cl_program program = build_program(context, device, kernel_src);
    cl_int ker_err;
    cl_kernel kernel = clCreateKernel(program, "yespower_smix", &ker_err);
    CHECK("clCreateKernel", ker_err);

    /* Verify initial S-boxes not needed - GPU handles S-init */

    /* 5. Create buffers */
    cl_int buf_err;
    /* Bi buffer - 1 work item */
    cl_mem d_Bi = clCreateBuffer(context, CL_MEM_READ_WRITE, 4096, NULL, &buf_err);
    CHECK("clCreateBuffer d_Bi", buf_err);

    /* V buffer - 8MB */
    size_t v_words = 2048 * 1024;
    size_t v_size = v_words * 4;
    cl_mem d_V = clCreateBuffer(context, CL_MEM_READ_WRITE, v_size, NULL, &buf_err);
    CHECK("clCreateBuffer d_V", buf_err);

    /* S buffer - 98304 bytes (kernel initializes it) */
    cl_mem d_S = clCreateBuffer(context, CL_MEM_READ_WRITE, Sbytes, NULL, &buf_err);
    CHECK("clCreateBuffer d_S", buf_err);

    /* Upload raw Bi (PBKDF2 output, GPU does S-init) */
    clEnqueueWriteBuffer(queue, d_Bi, CL_TRUE, 0, 4096, Bi, 0, NULL, NULL);

    /* Set kernel args */
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_Bi);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_V);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_S);
    uint32_t start_wg = 0, num_wgs = 1;
    clSetKernelArg(kernel, 3, sizeof(uint32_t), &start_wg);
    clSetKernelArg(kernel, 4, sizeof(uint32_t), &num_wgs);

    /* Run kernel */
    size_t global_size = 1;
    cl_event kernel_event;
    cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &kernel_event);
    CHECK("clEnqueueNDRangeKernel", err);

    clWaitForEvents(1, &kernel_event);
    printf("GPU kernel done\n");

    /* Read back Bi */
    uint8_t Bi_gpu[4096];
    clEnqueueReadBuffer(queue, d_Bi, CL_TRUE, 0, 4096, Bi_gpu, 0, NULL, NULL);

    print_hash("GPU post-smix Bi[0..7]", Bi_gpu, 8);
    print_hash("GPU post-smix Bi[4032..]", Bi_gpu + 4032, 32);

    /* 6. Compute final HMAC on CPU using GPU's Bi */
    uint8_t gpu_result[32];
    HMAC_SHA256_Buf(Bi_gpu + 4096 - 64, 64, key, 32, gpu_result);

    print_hash("GPU final hash", gpu_result, 32);
    print_hash("CPU reference hash", reference.uc, 32);

    /* Compare */
    if (memcmp(gpu_result, reference.uc, 32) == 0) {
        printf("\n✓ MATCH! GPU smix is correct!\n");
    } else {
        printf("\n✗ MISMATCH! GPU smix is wrong\n");
    }

    /* Cleanup */
    clReleaseMemObject(d_Bi);
    clReleaseMemObject(d_V);
    clReleaseMemObject(d_S);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(kernel_src);

    return 0;
}
