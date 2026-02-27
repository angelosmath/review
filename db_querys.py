import pandas as pd
import re
import matplotlib.pyplot as plt



# --------------------
# Load Data
# --------------------
pub = pd.read_csv("csv-KnowledgeG-set.csv")
scopus = pd.read_csv("scopus_export_Feb 25-2026_d93c044b-5fec-4da5-8d30-ad7cb2501782.csv")
ieee = pd.read_csv("export2026.02.25-06.45.35.csv")


# --------------------
# Helpers
# --------------------
def norm_text(x):
    if pd.isna(x):
        return "" # treat NaN as empty string for text normalization
    x = str(x).lower().strip() 
    x = re.sub(r"\s+", " ", x) 
    return x

def norm_doi(x):
    if pd.isna(x):
        return "" # treat NaN as empty string for DOI normalization
    x = str(x).strip().lower()
    x = x.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi:", "").strip()
    x = re.sub(r"\s+", "", x) 
    return x

def norm_title_key(title):
    t = norm_text(title)
    t = re.sub(r"[^\w\s]", " ", t)     # remove punctuation / removes avery thing that is not a letter like numbers, whitespaces etc. and replaces it with a space
    t = re.sub(r"\s+", " ", t).strip() # after cleaning, collapse multiple spaces and trim
    return t

def contains_any(text, keywords):
    t = norm_text(text)
    return any(k in t for k in keywords) # filtering if a keyword is present in a string, returns true if any of the keywords are present in the string


# ----------------------------------------------------
# Standardize each dataset to common columns
# ----------------------------------------------------


COMMON_COLUMNS = ["Title", "Authors", "Year", "Journal", "DOI", "Source"]

pub_df = pd.DataFrame({
    "Title": pub.get("Title", ""),
    "Authors": pub.get("Authors", ""),
    "Year": pub.get("Publication Year", ""),
    "Journal": pub.get("Journal/Book", ""),
    "DOI": pub.get("DOI", ""),
    "Source": "PubMed"
})

scopus_df = pd.DataFrame({
    "Title": scopus.get("Title", ""),
    "Authors": scopus.get("Author full names", ""),
    "Year": scopus.get("Year", ""),
    "Journal": scopus.get("Source title", ""),
    "DOI": scopus.get("DOI", ""),
    "Source": "Scopus"
})

ieee_df = pd.DataFrame({
    "Title": ieee.get("Document Title", ""),
    "Authors": ieee.get("Authors", ""),
    "Year": ieee.get("Publication Year", ""),
    "Journal": ieee.get("Publication Title", ""),
    "DOI": ieee.get("DOI", ""),
    "Source": "IEEE"
})

combined = pd.concat([pub_df, scopus_df, ieee_df], ignore_index=True)[COMMON_COLUMNS]


# --------------------
# Normalized fields
# --------------------
combined["DOI_norm"] = combined["DOI"].apply(norm_doi) # normalizing the DOI field by applying the norm_doi function to clean and standardize the DOI values across records
combined["Title_norm"] = combined["Title"].apply(norm_text) # normalizing the Title field by applying the norm_text function to clean and standardize the title values across records (lowercasing, trimming, collapsing spaces)
combined["Title_key"]  = combined["Title"].apply(norm_title_key) # creating a Title_key field by applying the norm_title_key function to the Title field, which further processes the title by removing punctuation and non-word characters, collapsing multiple spaces, and trimming. This creates a simplified version of the title that can be used for deduplication and matching purposes.
combined["Year"] = combined["Year"].astype(str).str.extract(r"(\d{4})", expand=False).fillna("") # extracting the year from the Year field by using a regular expression to find a 4-digit number, converting it to string, and filling any missing values with an empty string. This standardizes the Year field for easier comparison and analysis.

print("Combined dataset shape:", combined.shape)


counts = combined["Source"].value_counts().sort_index()

print("\nNumber of papers per database:")
print(counts)



# ----------------------------------------------------
# plot the standirized counts of papers per database as a bar chart 
# ----------------------------------------------------


plt.figure()

ax = counts.plot(kind="bar")

plt.title("Number of papers per database")
plt.xlabel("Database")
plt.ylabel("Number of papers")

plt.xticks(rotation=0)

# add value labels
for i, v in enumerate(counts):
    ax.text(i, v + 2, str(v), ha="center")

plt.tight_layout()
plt.show()

plt.savefig("01_records_per_database.png", dpi=300)




# -------------------------------------------------------------------------
# Flags: review / retracted / access
# -------------------------------------------------------------------------


# --------------------
# Review detection 
# --------------------

RE_DOCTYPE_REVIEW = re.compile(r"\breview\b", re.IGNORECASE)

RE_TITLE_REVIEW = re.compile(
    r"""
    \b(
        systematic\s+review|
        scoping\s+review|
        literature\s+review|
        narrative\s+review|
        umbrella\s+review|
        meta[-\s]?analysis|
        survey|
        state[-\s]?of[-\s]?the[-\s]?art\s+review|
        review
    )\b
    """,
    re.IGNORECASE | re.VERBOSE
)

def is_review_doctype(x):
    t = norm_text(x)
    return bool(RE_DOCTYPE_REVIEW.search(t))

def is_review_title(x):
    t = norm_text(x)
    return bool(RE_TITLE_REVIEW.search(t))

def review_reason(row):
    if is_review_doctype(row.get("DocType_raw", "")):
        return "DocType"
    if is_review_title(row.get("Title", "")):
        return "Title"
    if is_review_title(row.get("Citation", "")):
        return "Citation"
    return ""

combined["flag_review"] = combined.apply(lambda r: review_reason(r) != "", axis=1)
combined["flag_review_reason"] = combined.apply(review_reason, axis=1)


print(combined["flag_review"].value_counts())

print(
    combined.loc[
        combined["flag_review"],
        ["Title","Source","flag_review_reason"]
    ].head(20)
)


combined.to_csv("combined_all_with_flags.csv", index=False, encoding="utf-8-sig")

print("Total combined:", len(combined))
print("Total unique:", len(master_unique))

print(
    combined.loc[
        combined["Title"].str.contains("drug discovery platforms", case=False, na=False),
        ["Title","DOI","DOI_norm","dedup_key","flag_duplicate"]
    ]
)
breakpoint()

# Retract flags
retract_kw = ["retracted", "retraction", "withdrawn", "withdrawal", "retract"]
combined["flag_retracted"] = (
    combined["PublicationStage_raw"].apply(lambda x: contains_any(x, retract_kw)) |
    combined["Title"].apply(lambda x: contains_any(x, retract_kw)) |
    combined["Citation"].apply(lambda x: contains_any(x, retract_kw))
)

# Access flags 
combined["flag_prob_open_access"] = False
combined.loc[combined["Source"].eq("Scopus"), "flag_prob_open_access"] = combined.loc[
    combined["Source"].eq("Scopus"), "OpenAccess_raw"
].apply(lambda x: norm_text(x) not in ["", "no", "none", "closed", "non-open access"])

combined.loc[combined["Source"].eq("PubMed"), "flag_prob_open_access"] = combined.loc[
    combined["Source"].eq("PubMed"), "PMCID"
].apply(lambda x: norm_text(x) != "")

combined.loc[combined["Source"].eq("IEEE"), "flag_prob_open_access"] = combined.loc[
    combined["Source"].eq("IEEE"), "PDFLink"
].apply(lambda x: norm_text(x) != "")

combined["flag_prob_no_access"] = ~combined["flag_prob_open_access"]

# --------------------
# Dedup keys
# --------------------
combined["dedup_key"] = combined["DOI_norm"]
missing_doi = combined["dedup_key"].eq("")
combined.loc[missing_doi, "dedup_key"] = combined.loc[missing_doi].apply(
    lambda r: f"T:{r['Title_key']}|Y:{r['Year']}", axis=1
)

# Group duplicates + keep-first
combined["dup_group_size"] = combined.groupby("dedup_key")["dedup_key"].transform("size")
combined["flag_duplicate"] = combined["dup_group_size"].gt(1)

# Keep the "best" record per group:
# preference: has DOI > has abstract-ish info > has OA
combined["has_doi"] = combined["DOI_norm"].ne("")
combined["has_pdf"] = combined["PDFLink"].apply(lambda x: norm_text(x) != "")
combined["score_keep"] = combined["has_doi"].astype(int)*3 + combined["has_pdf"].astype(int)*2 + combined["flag_prob_open_access"].astype(int)

combined_sorted = combined.sort_values(["dedup_key", "score_keep"], ascending=[True, False])
master_unique = combined_sorted.drop_duplicates(subset="dedup_key", keep="first").copy()

# --------------------
# Outputs
# --------------------
combined.to_csv("combined_all_with_flags.csv", index=False)
master_unique.to_csv("master_unique_for_screening.csv", index=False)

print("All rows:", len(combined))
print("Unique after dedup:", len(master_unique))
print("Duplicates flagged:", int(combined["flag_duplicate"].sum()))
print("Review flagged (unique):", int(master_unique["flag_review"].sum()))
print("Retracted flagged (unique):", int(master_unique["flag_retracted"].sum()))
print("Prob no access (unique):", int(master_unique["flag_prob_no_access"].sum()))
print("Saved: combined_all_with_flags.csv, master_unique_for_screening.csv")



total_records = len(combined)
unique_after_dedup = len(master_unique)
removed_duplicates = total_records - unique_after_dedup

review_unique = int(master_unique["flag_review"].sum())
no_access_unique = int(master_unique["flag_prob_no_access"].sum())

labels = [
    "Total records",
    "Unique after dedup",
    "Removed duplicates",
    "Review (unique)",
    "No access (unique)"
]
values = [
    total_records,
    unique_after_dedup,
    removed_duplicates,
    review_unique,
    no_access_unique
]

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.ylabel("Count")
plt.title("PRISMA Identification Summary")
plt.xticks(rotation=20, ha="right")

for i, v in enumerate(values):
    plt.text(i, v, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.show()

print("\n=== PRISMA-style counts ===")
print("Total records:", total_records)
print("Unique after dedup:", unique_after_dedup)
print("Removed duplicates:", removed_duplicates)
print("Review (unique):", review_unique)
print("No access (unique):", no_access_unique)