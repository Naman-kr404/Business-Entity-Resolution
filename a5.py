import pandas as pd
import re
import json
import os
import time
import subprocess
import urllib.request

from collections import defaultdict
from rapidfuzz import fuzz
import ollama


# ============================================================
# PATHS
# ============================================================

BASE = "/Users/namankumar/Downloads/student_resource"

TRAIN_DIR = f"{BASE}/dataset/train"

S1_FILE = f"{TRAIN_DIR}/train_source1.tsv"
S2_FILE = f"{TRAIN_DIR}/train_source2.tsv"
S3_FILE = f"{TRAIN_DIR}/train_source3.tsv"

OUTPUT_FILE = f"{BASE}/matching_results_train.tsv"

ADDRESS_CACHE_FILE = f"{BASE}/address_match_cache.json"


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "qwen2.5:7b"

NAME_THRESHOLD = 72

PREFIX_LENGTH = 3

# Maximum number of name candidates sent to Qwen
TOP_K = 3


# ============================================================
# START OLLAMA SERVER IF NOT ALREADY RUNNING
# ============================================================

ollama_process = None

try:
    urllib.request.urlopen(
        "http://localhost:11434/api/tags",
        timeout=3
    )

    print("Ollama server is already running.")

except Exception:

    print("Starting Ollama server...")

    ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    time.sleep(5)

    print("Ollama server started.")


# ============================================================
# COMPANY SUFFIXES
# ============================================================

COMPANY_SUFFIXES = {
    "limited", "ltd",
    "private", "pvt", "priv",
    "company", "co",
    "corporation", "corp",
    "incorporated", "inc",
    "llc", "llp"
}


# ============================================================
# NORMALIZE COMPANY NAME
# ============================================================

def normalize_name(name):

    if pd.isna(name):
        return ""

    name = str(name).lower().strip()

    name = name.replace("&", " and ")
    name = name.replace("-", " ")

    name = re.sub(
        r"[^a-z0-9\s]",
        "",
        name
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    ).strip()

    words = name.split()

    while words and words[-1] in COMPANY_SUFFIXES:
        words.pop()

    return " ".join(words)


# ============================================================
# NORMALIZE ADDRESS
# ============================================================

def normalize_address(address):

    if pd.isna(address):
        return ""

    address = str(address).lower().strip()

    address = address.replace("&", " and ")
    address = address.replace("-", " ")

    address = re.sub(
        r"[^a-z0-9\s]",
        "",
        address
    )

    address = re.sub(
        r"\s+",
        " ",
        address
    ).strip()

    return address


# ============================================================
# SORT WORDS IN COMPANY NAME
# ============================================================

def sort_words(text):

    if not text:
        return ""

    words = text.split()
    words.sort()

    return " ".join(words)


# ============================================================
# CLEAN ENTITY ID
# ============================================================

def clean_entity_id(entity_id):

    if pd.isna(entity_id):
        return ""

    return str(entity_id).strip()


# ============================================================
# READ SOURCE FILES
# ============================================================

print("Reading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    dtype="string",
    keep_default_na=False
)


print("Reading Source 2...")

s2 = pd.read_csv(
    S2_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    dtype="string",
    keep_default_na=False
)


print("Reading Source 3...")

s3 = pd.read_csv(
    S3_FILE,
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country"
    ],
    dtype="string",
    keep_default_na=False
)


print("\nSource 1 rows:", len(s1))
print("Source 2 rows:", len(s2))
print("Source 3 rows:", len(s3))


# ============================================================
# NORMALIZE ALL SOURCES
# ============================================================

print("\nNormalizing company names and addresses...")

for df in [s1, s2, s3]:

    df["entity_id"] = (
        df["entity_id"].map(clean_entity_id)
    )

    df["normalized_name"] = (
        df["business_name"].map(normalize_name)
    )

    df["sorted_name"] = (
        df["normalized_name"].map(sort_words)
    )

    df["normalized_address"] = (
        df["business_address"].map(normalize_address)
    )


# ============================================================
# SORT SOURCE 1 BY COMPANY NAME
# ============================================================

print("\nSorting Source 1...")

s1_valid = s1[
    s1["sorted_name"] != ""
].copy()


s1_valid = s1_valid.sort_values(
    by="sorted_name",
    kind="mergesort"
).reset_index(drop=True)


# ============================================================
# SOURCE 1 ARRAYS
# ============================================================

s1_entity_ids = s1_valid["entity_id"].tolist()

s1_names = s1_valid["sorted_name"].tolist()

s1_addresses = s1_valid["business_address"].tolist()

s1_normalized_addresses = (
    s1_valid["normalized_address"].tolist()
)


print(
    "Valid Source 1 entities:",
    len(s1_valid)
)


# ============================================================
# BUILD PREFIX INDEX USING COMPANY NAME ONLY
# ============================================================

print("\nBuilding company-name prefix index...")

prefix_index = defaultdict(list)

for position, name in enumerate(s1_names):

    prefix = name[:PREFIX_LENGTH]

    prefix_index[prefix].append(position)


print(
    "Number of prefix groups:",
    len(prefix_index)
)


# ============================================================
# GET CANDIDATES USING COMPANY NAME PREFIX
# ============================================================

def get_name_candidates(query_name):

    if not query_name:
        return []

    prefix = query_name[:PREFIX_LENGTH]

    return prefix_index.get(
        prefix,
        []
    )


# ============================================================
# LOAD ADDRESS MATCH CACHE
# ============================================================

if os.path.exists(ADDRESS_CACHE_FILE):

    with open(
        ADDRESS_CACHE_FILE,
        "r"
    ) as f:

        address_match_cache = json.load(f)

    print(
        "\nLoaded address cache:",
        len(address_match_cache)
    )

else:

    address_match_cache = {}


# ============================================================
# QWEN ADDRESS COMPARISON
# ============================================================


def compare_addresses(address1, address2):

    normalized1 = normalize_address(address1)
    normalized2 = normalize_address(address2)

    if not normalized1 or not normalized2:
        return False

    # Exact address match does not require Qwen.
    if normalized1 == normalized2:
        return True

    # Create order-independent cache key.
    pair = sorted([
        normalized1,
        normalized2
    ])

    cache_key = json.dumps(pair)

    if cache_key in address_match_cache:
        return address_match_cache[cache_key]

    prompt = f"""
    You are an expert business address matching system.

    Determine whether the following two addresses refer to
    the SAME general business location.

    ADDRESS 1:
    {address1}

    ADDRESS 2:
    {address2}

    MATCHING RULES:

    1. Ignore differences in capitalization, punctuation,
       spacing, and common street abbreviations.

    2. Ignore differences or errors in:
       - House numbers
       - Building numbers
       - Street numbers
       - Street names
       - Floor numbers
       - Landmark names
       - Minor locality details

    3. If one address contains more details than the other,
       they can still be considered the same address.

       Example:
       "123, Boring Road, Patna, Bihar"
       "Patna, Bihar"
       Result: SAME LOCATION.

    4. If one address contains a landmark, street name,
       or building detail that is missing from the other,
       do not reject the match just because of the
       additional information.

    5. Prioritize matching the available broader location
       information, such as:
       - City
       - State or province
       - Country
       - Postal code, when available

    6. If the city, state, or country clearly conflicts
       between the two addresses, return false.

    7. Missing location details should not automatically
       be treated as conflicting information.

    8. Do not require exact agreement on house numbers,
       building numbers, street names, or landmarks.

    9. If the available information supports the same
       general business location, return true.

    10. If there is insufficient information to establish
        a match, return false.

    Return valid JSON only:

    {{
        "same_location": false,
        "confidence": 0
    }}
    """

    try:

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0
            }
        )

        result = json.loads(
            response.message.content
        )

        same_location = result.get(
            "same_location",
            False
        )

        confidence = float(
            result.get("confidence", 0)
        )

        if isinstance(same_location, str):
            same_location = (
                same_location.lower() == "true"
            )

        is_match = (
            same_location is True
            and confidence >= 80
        )

    except Exception as e:

        print(
            "Address comparison error:",
            e
        )

        is_match = False

    address_match_cache[cache_key] = is_match

    return is_match



# ============================================================
# FIND MATCH IN SOURCE 1
# ============================================================

def find_best_match(query_name, query_address):

    if not query_name:
        return None

    # --------------------------------------------------------
    # STEP 1: Find candidates using company name prefix
    # --------------------------------------------------------

    candidate_positions = get_name_candidates(
        query_name
    )

    if not candidate_positions:
        return None

    # --------------------------------------------------------
    # STEP 2: Apply RapidFuzz to company names
    # --------------------------------------------------------

    scored_candidates = []

    for position in candidate_positions:

        name_score = fuzz.ratio(
            query_name,
            s1_names[position]
        )

        if name_score >= NAME_THRESHOLD:

            scored_candidates.append(
                (
                    name_score,
                    position
                )
            )

    # Sort by company-name similarity

    scored_candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # --------------------------------------------------------
    # STEP 3: Compare addresses only when company names match
    # --------------------------------------------------------

    for name_score, position in scored_candidates[:TOP_K]:

        candidate_address = s1_addresses[position]

        if not query_address or not candidate_address:
            continue

        is_match = compare_addresses(
            query_address,
            candidate_address
        )

        if is_match:

            return (
                position,
                name_score
            )

    return None


# ============================================================
# PROCESS SOURCE 2 / SOURCE 3
# ============================================================

def process_source(source_df, source_name):

    print(
        f"\n========================================"
    )

    print(
        f"Processing {source_name}"
    )

    print(
        f"========================================"
    )

    entity_ids = source_df["entity_id"].tolist()

    names = source_df["sorted_name"].tolist()

    addresses = source_df["business_address"].tolist()

    total = len(entity_ids)

    results = {}

    # Cache repeated source records

    cache = {}

    cache_hits = 0
    match_count = 0

    for row_index in range(total):

        query_name = names[row_index]

        query_address = addresses[row_index]

        cache_key = (
            query_name,
            normalize_address(query_address)
        )

        if cache_key in cache:

            matched = cache[cache_key]

            cache_hits += 1

        else:

            matched = find_best_match(
                query_name,
                query_address
            )

            cache[cache_key] = matched

        if matched is not None:

            s1_position, name_score = matched

            s1_id = s1_entity_ids[s1_position]

            results[row_index] = s1_id

            match_count += 1

        else:

            results[row_index] = None

        # Save address cache periodically

        if (row_index + 1) % 1000 == 0:

            with open(
                ADDRESS_CACHE_FILE,
                "w"
            ) as f:

                json.dump(
                    address_match_cache,
                    f
                )

        if (
            (row_index + 1) % 1000 == 0
            or row_index + 1 == total
        ):

            print(
                f"{source_name}: "
                f"{row_index + 1:,}/{total:,} processed | "
                f"matches={match_count:,} | "
                f"cache_hits={cache_hits:,}"
            )

    print(
        f"\n{source_name} matches:",
        match_count
    )

    print(
        f"{source_name} cache hits:",
        cache_hits
    )

    return results


# ============================================================
# MATCH SOURCE 2
# ============================================================

s2_results = process_source(
    s2,
    "Source 2"
)


with open(
    ADDRESS_CACHE_FILE,
    "w"
) as f:

    json.dump(
        address_match_cache,
        f
    )


# ============================================================
# MATCH SOURCE 3
# ============================================================

s3_results = process_source(
    s3,
    "Source 3"
)


with open(
    ADDRESS_CACHE_FILE,
    "w"
) as f:

    json.dump(
        address_match_cache,
        f
    )


# ============================================================
# CREATE MATCH DICTIONARY
# ============================================================

matches = defaultdict(list)


# ============================================================
# ADD SOURCE 2 MATCHES
# ============================================================

s2_entity_ids = s2["entity_id"].tolist()

for row_index, s1_id in s2_results.items():

    if s1_id is None:
        continue

    s2_id = s2_entity_ids[row_index]

    matches[s1_id].append(s2_id)


# ============================================================
# ADD SOURCE 3 MATCHES
# ============================================================

s3_entity_ids = s3["entity_id"].tolist()

for row_index, s1_id in s3_results.items():

    if s1_id is None:
        continue

    s3_id = s3_entity_ids[row_index]

    matches[s1_id].append(s3_id)


# ============================================================
# CREATE OUTPUT
# ============================================================

output = []

for s1_id in s1["entity_id"]:

    matched = matches.get(
        s1_id,
        []
    )

    unique_ids = list(
        dict.fromkeys(matched)
    )

    output.append(
        {
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(unique_ids)
        }
    )


# ============================================================
# SAVE OUTPUT
# ============================================================

result_df = pd.DataFrame(
    output,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)


result_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

with_matches = (
    result_df["matched_entity_ids"] != ""
).sum()

without_matches = (
    result_df["matched_entity_ids"] == ""
).sum()


print("\n========================================")
print("DONE")
print("========================================")

print("Output:", OUTPUT_FILE)

print("Rows:", len(result_df))

print("S1 entities with matches:", with_matches)

print("S1 entities without matches:", without_matches)

print("Name threshold:", NAME_THRESHOLD)

print("Prefix length:", PREFIX_LENGTH)

print("Top K:", TOP_K)

print(
    "Address comparisons cached:",
    len(address_match_cache)
)

print("\nSample output:")

print(
    result_df.head(10).to_string(index=False)
)