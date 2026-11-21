# Rust Shopping Sessions pipeline

Build from this directory with Docker.

```sh
docker build --platform linux/amd64 -t shopping-rust .
```

Run with the challenge input mounted at `/data` and a directory writable by UID 1000 at `/output`. The executable also accepts input and output paths as its first two arguments. `NCPU` controls the Rayon worker count and defaults to eight. Set `SHOP_PROFILE=1` to report phase timings to stderr.

Workers scan memory-mapped CSV chunks on complete-line boundaries. Identity links are reconciled concurrently with event parsing. Events are partitioned by ID. A bitmap identifies hash slots containing a single record, which can pass directly to the person partition. Repeated slots receive exact ID comparisons and earliest-ingest deduplication. Invalid earliest copies remain eligible to suppress later copies.

Each person partition groups and sorts its timestamps. A sliding seven-day click window determines session credit and distinct assisting campaigns. Identical click copies enter and expire together, so they produce the same credit and active campaign set as a single copy. The Rust Parquet writer emits one user partition per worker task with rows and merges campaign counts into one file.

Example run from a Linux directory containing the challenge inputs in `data/`. Give UID 1000 write access to `output/` first.

```sh
docker run --rm --platform linux/amd64 --network none --read-only \
  --cpus 8 --memory 24g --memory-swap 24g \
  -v "$PWD/data:/data:ro" -v "$PWD/output:/output" shopping-rust
```

## Local comparison

Using the official generator with seed 29, 20 million events and five million clicks, three fresh containers completed in 1.143, 1.129 and 1.050 seconds (median **1.129 s**). The input volume was warm, eight logical CPUs were pinned, the container memory limit was 7 GiB, and swap was disabled. Image order rotated against the scored C++ commit, whose local median was 1.262 seconds, and the Rust full-table dedup version, whose median was 1.229 seconds.

Both output tables matched all 36 fixtures and the official 1M-event dev answers. The practice run matched every row of the validated reference output (2,240,907 user rows and 28,014 campaign rows). These timings were measured on the local Docker host.
