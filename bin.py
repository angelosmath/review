

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




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



# ============================================================
# ENTRY POINT 
# ============================================================
if __name__ == "__main__":
    identifier = LiteratureIdentification(
        pubmed_path = "csv-KnowledgeG-set.csv",
        scopus_path = "scopus_export_Feb_25-2026_d93c044b-5fec-4da5-8d30-ad7cb2501782.csv",
        ieee_path   = "export2026_02_25-06_45_35.csv"
    )
    master = identifier.run()