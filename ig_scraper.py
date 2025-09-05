#!/usr/bin/env python3
"""
Fetch Instagram posts (captions + metadata) via Apify's 'Instagram Post Scraper'.

Usage examples:
  python3 ig_scraper.py --usernames natgeo nasa --limit 200 --cutoff "2025-01-01" --out captions.csv
  pytho3n ig_scraper.py --urls https://www.instagram.com/p/DLNsnpUTdVS/ --out json

Notes:
- Provide either --usernames (space-separated) OR --urls (space-separated post/profile URLs).
- --out with 'csv' or 'json' determines the export format (defaults to CSV).
"""

import os
import argparse
from typing import List, Optional
from apify_client import ApifyClient

ACTOR_ID = "apify/instagram-post-scraper"  # official actor

def run_scraper(
    token: str,
    usernames: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    limit_per_profile: Optional[int] = None,
    cutoff: Optional[str] = None,
    skip_pinned: bool = False,
):
    client = ApifyClient(token)

    # Build input according to the actor's schema
    # Docs show: username (array), resultsLimit (int), onlyPostsNewerThan (string), skipPinnedPosts (bool)
    # You can pass either usernames or direct post/profile URLs in the same "username" array.
    targets: List[str] = []
    if usernames:
        targets.extend(usernames)
    if urls:
        targets.extend(urls)
    if not targets:
        raise SystemExit("Provide --usernames or --urls")

    run_input = {
        "username": targets,
    }
    if limit_per_profile is not None:
        run_input["resultsLimit"] = int(limit_per_profile)
    if cutoff:
        run_input["onlyPostsNewerThan"] = cutoff  # e.g., "2025-01-01" or "2 months"
    if skip_pinned:
        run_input["skipPinnedPosts"] = True

    # Start the actor and wait for it to finish
    run = client.actor(ACTOR_ID).call(run_input=run_input)

    # Get dataset id + return a dataset client
    dataset_id = run["defaultDatasetId"]
    ds = client.dataset(dataset_id)
    return ds

def export_dataset(ds, out_path: str):
    import json, pathlib
    items = list(ds.iterate_items())
    pathlib.Path(out_path).write_text(
        json.dumps(items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Wrote JSON: {out_path}  (items={len(items)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usernames", nargs="*", help="Instagram usernames (no @), e.g. natgeo nasa")
    parser.add_argument("--urls", nargs="*", help="Instagram post/profile URLs")
    parser.add_argument("--limit", type=int, default=None, help="Max posts per profile")
    parser.add_argument("--cutoff", type=str, default=None, help='Cutoff, e.g. "2025-01-01" or "2 months"')
    parser.add_argument("--skip-pinned", action="store_true", help="Skip pinned posts")
    parser.add_argument("--out", default="captions.csv", help="Output file path; .csv or .json")
    args = parser.parse_args()

    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise SystemExit("Set APIFY_TOKEN environment variable.")

    ds = run_scraper(
        token=token,
        usernames=args.usernames,
        urls=args.urls,
        limit_per_profile=args.limit,
        cutoff=args.cutoff,
        skip_pinned=args.skip_pinned,
    )

    export_dataset(ds, args.out)


if __name__ == "__main__":
    main()
