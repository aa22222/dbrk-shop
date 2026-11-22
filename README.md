# Shopping Sessions

C++20 implementation for the Shopping Sessions challenge. The executable reconciles linked user IDs, deduplicates and sessionizes events, attributes sessions to campaigns, and writes daily Parquet tables with Apache Arrow.

## Build and run

Build the Linux amd64 image from the repository root.

```sh
docker build --platform linux/amd64 -t shopping-sessions .
```

From a directory containing the challenge input in `data/`, prepare an empty output directory writable by uid 1000, then run the container.

```sh
mkdir -p output
docker run --rm --platform linux/amd64 --network none --read-only \
  --cpus 8 --memory 24g --memory-swap 24g \
  -v "$PWD/data:/data:ro" -v "$PWD/output:/output" shopping-sessions
```

The two tables appear in `output/user_daily/` and `output/campaign_daily/`. The executable reads `DATA_DIR`, `OUTPUT_DIR`, and `NCPU`, which default to `/data`, `/output`, and `8`. Set `SHOP_PROFILE=1` in the container to log stage timings and peak RSS.

## Array layout

Workers decode canonical base-36 event IDs into integers. Low digits select one of 64 shards and a byte counter within that shard. Each counter stops at two. A count of one proves the event ID is unique. Workers sort records sharing a counter by their full IDs and keep the earliest ingestion time for each exact ID. Other event-ID spellings use the same sort on the original CSV field bytes. An invalid earliest copy still suppresses later copies.

Shopper lookups use digit-indexed ranges in sorted arrays. Linked people have dense ranks within each shard. Product and campaign numbers, UTC day offsets, and session totals also index arrays. Ordered maps handle unusual string identifiers and catalog entries.

## Local practice benchmark

The official generator produced 20 million events and five million clicks with seed 29. Each cell is the median start-to-exit time of five fresh containers. Run order rotated across builds. Each container had eight pinned logical CPUs, 7 GiB of memory and zero swap. Input page residency was checked before each run.

| Build | Warm input | Cold input |
| --- | ---: | ---: |
| This C++ array implementation | 1.092 s | 1.343 s |
| Previous C++ (`6c58378`) | 1.085 s | 1.387 s |
| Rust (`0cee854`) | 0.913 s | 1.100 s |

The C++ array implementation peaked at 3,115 MiB RSS across these runs. Both outputs matched the 36 fixtures, the official one-million-event dev answers, and every row of the practice reference. A separate check deliberately placed 4,096 distinct event IDs in the same counter slot and verified their outcomes independently.
