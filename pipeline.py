"""Strict native CSV ingestion with Polars deduplication, sessions and attribution."""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", os.environ.get("NCPU", "8"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import polars as pl
import shop_reader

WEEK = 604_800_000
USER = {
    "user_id": pl.String, "day": pl.Date, "sessions": pl.UInt32,
    "events": pl.UInt32, "revenue_cents": pl.Int64, "top_category": pl.String,
}
CAMPAIGN = {
    "campaign_id": pl.String, "channel": pl.String, "day": pl.Date,
    "sessions": pl.UInt32, "purchasing_sessions": pl.UInt32,
    "revenue_cents": pl.Int64, "assisted_sessions": pl.UInt32,
}
C = pl.col


def deduplicate(reader, person_shards, event_batches):
    by_person = [[] for _ in range(person_shards)]
    width = shop_reader.SHARDS // event_batches
    for first in range(0, shop_reader.SHARDS, width):
        candidates = pl.concat([pl.DataFrame(reader.event_part(b)) for b in range(first, first+width)])
        if not candidates.height:
            continue
        # Invalid copies carry a negative price so they can still win dedup.
        clean = (
            candidates.sort("ingest").unique("id", keep="first")
            .filter(C("price") >= 0)
            .select((C("shard") % person_shards).alias("shard"), "user", "t", "price", "category")
        )
        for (owner,), part in clean.partition_by("shard", as_dict=True, include_key=False).items():
            by_person[owner].append(part)
    return by_person


def session_tables(events):
    ordered = events.sort("user", "t")
    begins = (
        (C("user") != C("user").shift()) | (C("t").diff() > 1_800_000)
    ).fill_null(True)
    labeled = ordered.with_columns(begins.cast(pl.UInt32).cum_sum().set_sorted().alias("sid"))
    sessions = (
        labeled.group_by("sid", maintain_order=True).agg(C("user").first(), C("t").first().alias("start"),
            pl.len().alias("events"), C("price").cast(pl.Int64).sum().alias("revenue_cents"))
        .with_columns(pl.from_epoch("start", time_unit="ms").dt.date().alias("day"))
    )
    top = (
        labeled.filter(C("price") > 0).select("sid","user","category","price")
        .join(sessions.select("sid","day"),on="sid")
        .group_by("user", "day", "category").agg(C("price").cast(pl.Int64).sum().alias("spend"))
        .sort(["spend", "category"], descending=[True, False])
        .unique(["user", "day"], keep="first").select("user", "day", "category")
    )
    users = (
        sessions.group_by("user", "day").agg(
            pl.len().alias("sessions"), C("events").sum(), C("revenue_cents").sum()
        ).join(top, on=["user", "day"], how="left")
    )
    return sessions, users


def campaign_counts(sessions, clicks):
    # Collapse equal timestamps first. Repeated identical clicks cannot change
    # the winner, and overlapping eligibility intervals coalesce below.
    winners = (
        clicks.sort(["user", "t", "rank"], descending=[False,False,True])
        .select("user", "t", "campaign")
    )
    credited = (
        sessions.join_asof(winners, left_on="start", right_on="t", by="user",
                           strategy="backward", tolerance=WEEK, check_sortedness=False)
        .with_columns(C("campaign").fill_null(0))
        .group_by("campaign", "day").agg(
            pl.len().alias("sessions"),
            (C("revenue_cents") > 0).sum().alias("purchasing_sessions"),
            C("revenue_cents").sum(),
        )
    )
    ordered = clicks.select("user", "campaign", "t").sort("user", "campaign", "t")
    new_interval = (
        (C("user") != C("user").shift()) |
        (C("campaign") != C("campaign").shift()) |
        (C("t").diff() > WEEK)
    ).fill_null(True)
    intervals = (
        ordered.with_columns(new_interval.cast(pl.UInt32).cum_sum().set_sorted().alias("interval"))
        .group_by("interval").agg(
            C("user").first(), C("campaign").first(), C("t").min().alias("lo"),
            (C("t").max() + WEEK).alias("hi"),
        )
    )
    assisted = (
        sessions.select("user", "start", "day").lazy().join(intervals.lazy(), on="user")
        .filter(C("start").is_between(C("lo"), C("hi")))
        .group_by("campaign", "day").agg(pl.len().alias("assisted_sessions")).collect()
    )
    return credited.join(assisted, on=["campaign", "day"], how="full", coalesce=True).fill_null(0)


def run(data, out):
    started = time.perf_counter()
    def profile(name):
        nonlocal started
        now = time.perf_counter()
        if os.getenv("SHOP_PROFILE"):
            print(f"{name} {now-started:.4f} s", file=sys.stderr, flush=True)
        started = now
    window = json.loads((data/"manifest.json").read_text())
    epoch = lambda key: int(datetime.fromisoformat(window[key]).timestamp() * 1000)
    reader = shop_reader.Reader(str(data), epoch("window_start"), epoch("window_end"),
                                int(os.getenv("NCPU", "8")))
    profile("native ingest and identities")
    count = reader.event_count()
    batches = min(64, 1 << (max(1, count // 1_000_000) - 1).bit_length())
    person_shards = min(64, 1 << (max(1, count // 1_500_000) - 1).bit_length())
    parts = deduplicate(reader, person_shards, batches)
    profile("Polars event deduplication and repartition")
    users_path = out/"user_daily"
    campaigns_path = out/"campaign_daily"
    users_path.mkdir(parents=True, exist_ok=True)
    campaigns_path.mkdir(parents=True, exist_ok=True)
    categories = pl.DataFrame({
        "category": pl.Series(range(len(reader.categories)), dtype=pl.Int32),
        "top_category": pl.Series(reader.categories, dtype=pl.String),
    })
    daily = []
    for shard, fragments in enumerate(parts):
        if not fragments:
            continue
        events = pl.concat(fragments, rechunk=True)
        fragments.clear()
        sessions, users = session_tables(events)
        keys = users.get_column("user").unique()
        people = pl.DataFrame({"user": keys, "user_id": pl.Series(reader.decode(keys.to_numpy()), dtype=pl.String)})
        users.join(people, on="user").join(categories, on="category", how="left").select(*USER).write_parquet(
            users_path/f"{shard}.parquet", compression="uncompressed", statistics=False)
        clicks = pl.concat([
            pl.DataFrame(dict(zip(("user", "t", "campaign", "rank"), reader.click_part(b))))
            for b in range(shard, shop_reader.SHARDS, person_shards)
        ])
        daily.append(campaign_counts(sessions, clicks))
    profile("Polars sessions, attribution and user output")
    if daily:
        names = reader.campaigns
        dimension = pl.DataFrame({
            "campaign": pl.Series(range(len(names)), dtype=pl.Int32),
            "campaign_id": [x[0] for x in names], "channel": [x[1] for x in names],
        })
        totals = (
            pl.concat(daily).group_by("campaign", "day").agg(
                C("sessions").sum(), C("purchasing_sessions").sum(),
                C("revenue_cents").sum(), C("assisted_sessions").sum()
            ).join(dimension, on="campaign").select(*CAMPAIGN)
        )
    else:
        pl.DataFrame(schema=USER).write_parquet(users_path/"0.parquet")
        totals = pl.DataFrame(schema=CAMPAIGN)
    totals.write_parquet(campaigns_path/"0.parquet", compression="uncompressed", statistics=False)
    profile("Polars campaign output")


if __name__ == "__main__":
    run(Path(sys.argv[1] if len(sys.argv)>1 else os.getenv("DATA_DIR","/data")),
        Path(sys.argv[2] if len(sys.argv)>2 else os.getenv("OUTPUT_DIR","/output")))
