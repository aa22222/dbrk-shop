# Polars candidate

Build the standalone Linux image from the repository root.

```sh
docker build --platform linux/amd64 -t dbrk-polars .
docker run --rm --read-only --network none --cpus 8 --memory 7g --memory-swap 7g \
  -v "$PWD/data:/data:ro" -v "$PWD/output:/output" dbrk-polars
```

Create an empty output directory writable by uid 1000 before the run. The entrypoint accepts `DATA_DIR`, `OUTPUT_DIR`, and `NCPU`, or input and output paths as two positional arguments. `SHOP_PROFILE=1` prints stage timings to stderr.

The native extension reads the specified CSV dialect, decodes timestamps and IDs, and resolves the transitive identity graph. It passes typed arrays to Polars. Polars selects the earliest event copy before dropping invalid winners, groups by canonical person, sessionizes chronological events, computes category revenue, credits campaigns by as-of join, counts assists from merged eligibility intervals, and writes both Parquet tables. Event-ID partitions and person partitions bound the size of each Polars query. The reader stores parsed input in memory.

## Validation and timing

The Linux amd64 image matched both expected tables on all 36 supplied fixtures and the official 1M-event seed-1 development input (117,038 user-day rows and 27,400 campaign-day rows). On the supplied generator's 20M-event, 5M-click seed-29 workload, both tables matched the scored C++ implementation row for row (2,240,907 user-day rows and 28,014 campaign-day rows).

Three local Docker runs on that workload took 9.478, 7.684, and 7.578 seconds from container start to exit, with eight pinned logical CPUs, a 7 GiB memory cap and swap disabled. The operating system retained its input cache between runs. On the median run the native ingest and identity stage took 0.604 seconds, event conversion/deduplication/repartitioning took 2.867 seconds, and per-person processing and user Parquet output took 3.889 seconds. The campaign merge and write took 0.011 seconds. These are local measurements.
