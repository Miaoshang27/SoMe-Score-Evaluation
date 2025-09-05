import json
from pathlib import Path

# List of your JSON files
files = [
    "/Users/miao.shang/Downloads/dataset_instagram-post-scraper_2025-09-03_09-10-25-136.json",
    "/Users/miao.shang/Downloads/dataset_instagram-post-scraper_2025-09-03_09-04-29-313.json",
    "/Users/miao.shang/Downloads/dataset_instagram-post-scraper_2025-09-03_08-24-55-098.json",
    "/Users/miao.shang/Downloads/dataset_instagram-post-scraper_2025-09-03_08-22-45-909.json",
    "/Users/miao.shang/Downloads/dataset_instagram-post-scraper_2025-09-03_09-16-28-385.json",
    "/Users/miao.shang/Downloads/dataset_instagram-post-scraper_2025-09-03_08-19-44-431.json",
    # add all your files here...
]

all_items = []
for f in files:
    with open(f, "r", encoding="utf-8") as infile:
        data = json.load(infile)
        if isinstance(data, list):
            all_items.extend(data)
        else:
            all_items.append(data)

# Write merged file
out_file = "posts.json"
with open(out_file, "w", encoding="utf-8") as outfile:
    json.dump(all_items, outfile, ensure_ascii=False, indent=2)

print(f"Merged {len(files)} files → {len(all_items)} items written to {out_file}")
