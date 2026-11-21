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
