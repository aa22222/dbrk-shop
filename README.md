# Rust Shopping Sessions pipeline

Build from this directory with Docker.

```sh
docker build --platform linux/amd64 -t shopping-rust .
```

Run with the challenge input mounted at `/data` and a directory writable by UID 1000 at `/output`. The executable also accepts input and output paths as its first two arguments. `NCPU` controls the Rayon worker count and defaults to eight. Set `SHOP_PROFILE=1` to report phase timings to stderr.

Workers scan memory-mapped CSV chunks in 8 MiB batches on complete-line boundaries. Identity links are reconciled concurrently with event parsing. The timestamp parser decodes 13-digit epoch milliseconds and fixed-width clock fields in batches, with a checked decimal path for other epoch lengths. Events are partitioned by ID. A bitmap identifies hash slots containing a single record, which can pass directly to the person partition. Repeated slots receive exact ID comparisons and earliest-ingest deduplication. An invalid earliest copy suppresses later copies, even when they name another user.

When the manifest window fits in a 32-bit millisecond offset and the category dictionary fits in eight bits, each event timeline entry uses eight bytes. The entry contains the offset, the category, and the purchase price in cents. Wider windows or dictionaries select the full-width format at startup.

Each person partition groups and sorts its timestamps. A sliding seven-day click window determines session credit and distinct assisting campaigns. Identical click copies enter and expire together, so they produce the same credit and active campaign set as a single copy. The Rust Parquet writer emits one user partition per worker task with rows and merges campaign counts into one file.

Example run from a Linux directory containing the challenge inputs in `data/`. Give UID 1000 write access to `output/` first.

```sh
docker run --rm --platform linux/amd64 --network none --read-only \
  --cpus 8 --memory 24g --memory-swap 24g \
  -v "$PWD/data:/data:ro" -v "$PWD/output:/output" shopping-rust
```

## Local comparison

The official generator produced 20 million events and five million clicks with seed 29. Each time below is a median of five fresh containers measured from start to exit. Both builds used eight pinned logical CPUs, a 7 GiB memory limit, and zero swap. Runs alternated between builds. Input page residency was checked before every run.

| Input cache | Previous Rust (`e42788b`) | This version |
| --- | ---: | ---: |
| Warm | 1.038 s | 0.857 s |
| Cold | 1.265 s | 1.018 s |

Both output tables matched all 36 fixtures and the official 1M-event dev answers. The practice run matched every row of the validated reference output (2,240,907 user rows and 28,014 campaign rows). These timings were measured on the local Docker host.
