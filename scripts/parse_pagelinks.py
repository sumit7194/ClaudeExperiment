"""
parse_pagelinks.py — Parse Wikipedia pagelinks SQL dump into a title-based link JSON.

The pagelinks dump stores source pages by ID, targets by title.
To resolve source IDs to titles, also provide the page SQL dump.

Usage:
    python scripts/parse_pagelinks.py \\
        --pagelinks data/enwiki-20180401-pagelinks.sql.gz \\
        --page      data/enwiki-20180401-page.sql.gz \\
        --output    data/links_2018.json

Download dumps from:
    https://dumps.wikimedia.org/enwiki/<date>/enwiki-<date>-pagelinks.sql.gz
    https://dumps.wikimedia.org/enwiki/<date>/enwiki-<date>-page.sql.gz
"""

import argparse
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from tqdm import tqdm


def parse_page_sql(page_sql_gz_path):
    """
    Parse the Wikipedia page SQL dump to build a page_id -> title mapping.
    Only includes namespace-0 (main article) pages.

    Returns:
        dict: {page_id (int): title (str)}
    """
    id_to_title = {}
    # page table: (page_id, page_namespace, page_title, ...)
    pattern = re.compile(r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)'")

    print(f"Parsing page table from {page_sql_gz_path} ...")
    with gzip.open(page_sql_gz_path, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            for match in pattern.finditer(line):
                page_id = int(match.group(1))
                namespace = int(match.group(2))
                title = match.group(3).replace("\\'", "'").replace("_", " ")
                if namespace == 0:
                    id_to_title[page_id] = title

    print(f"Found {len(id_to_title):,} main-namespace articles.")
    return id_to_title


def parse_pagelinks_sql(pagelinks_sql_gz_path, id_to_title=None):
    """
    Parse the Wikipedia pagelinks SQL dump.

    The dump rows are:
        (source_page_id, target_namespace, 'target_title', source_namespace)

    If id_to_title is provided, resolves source IDs to titles.
    Otherwise, returns {source_id (int): set(target_title)}.

    Returns:
        dict: {source_title_or_id: set(target_title)}
    """
    links = defaultdict(set)
    # target title may contain escaped quotes
    pattern = re.compile(r"\((\d+),(\d+),'((?:[^'\\]|\\.)*)',(\d+)\)")
    row_count = 0

    print(f"Parsing pagelinks from {pagelinks_sql_gz_path} ...")
    with gzip.open(pagelinks_sql_gz_path, "rt", errors="replace") as f:
        for line in f:
            if not line.startswith("INSERT INTO"):
                continue
            for match in pattern.finditer(line):
                source_id = int(match.group(1))
                target_ns = int(match.group(2))
                target_title = match.group(3).replace("\\'", "'").replace("_", " ")
                # namespace 0 = main article space only
                if target_ns == 0:
                    if id_to_title is not None:
                        if source_id in id_to_title:
                            links[id_to_title[source_id]].add(target_title)
                    else:
                        links[source_id].add(target_title)
                    row_count += 1

    key_type = "title" if id_to_title is not None else "page_id"
    print(f"Parsed {row_count:,} links from {len(links):,} source pages ({key_type}-keyed).")
    return links


def load_link_set(links_json_path):
    """
    Load a title-keyed link JSON file.

    Returns:
        set of (source_lower, target_lower) tuples — bidirectional
    """
    with open(links_json_path) as f:
        link_dict = json.load(f)

    link_set = set()
    for source, targets in link_dict.items():
        s = source.lower()
        for target in targets:
            t = target.lower()
            link_set.add((s, t))
            link_set.add((t, s))

    return link_set


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pagelinks", required=True, help="Path to enwiki-*-pagelinks.sql.gz")
    parser.add_argument("--page", required=False, help="Path to enwiki-*-page.sql.gz (required for title-keyed output)")
    parser.add_argument("--output", required=True, help="Output JSON path, e.g. data/links_2018.json")
    args = parser.parse_args()

    id_to_title = None
    if args.page:
        id_to_title = parse_page_sql(args.page)
    else:
        print(
            "WARNING: --page not provided. Output will be keyed by page ID, not title.\n"
            "  find_hidden_connections.py requires title-keyed links.\n"
            "  Provide --page enwiki-*-page.sql.gz for correct behaviour."
        )

    links = parse_pagelinks_sql(args.pagelinks, id_to_title)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    serializable = {str(k): list(v) for k, v in links.items()}
    with open(args.output, "w") as f:
        json.dump(serializable, f)

    total_links = sum(len(v) for v in links.values())
    print(f"Saved {total_links:,} links to {args.output}")


if __name__ == "__main__":
    main()
