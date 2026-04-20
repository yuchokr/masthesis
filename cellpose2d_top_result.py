import pandas as pd

INPUT = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_hit_annotation_summary.csv"
OUT = "/data/gent/vo/000/gvo00070/vsc48277/yujin_project/40k_subset_0_result/doublet_candidate_groups.csv"

df = pd.read_csv(INPUT)

def parse_types(s):
    vals = []
    for x in str(s).split(";"):
        x = x.strip()
        if x:
            vals.append(x)
    return vals

groups = []
for _, r in df.iterrows():
    types = parse_types(r["unique_cell_types_across_z"])
    known = [t for t in types if "unknown" not in t.lower()]

    if len(known) >= 2 and not r["has_unknown_any_z"]:
        group = "clean_multitype"
    elif len(known) >= 2:
        group = "mixed_but_interesting"
    else:
        group = "mostly_unknown"

    groups.append(group)

df["candidate_group"] = groups
df.to_csv(OUT, index=False)

print(df["candidate_group"].value_counts())
print("\n=== clean_multitype ===")
print(df[df["candidate_group"] == "clean_multitype"].head(20).to_string(index=False))

print("\n=== mixed_but_interesting ===")
print(df[df["candidate_group"] == "mixed_but_interesting"].head(20).to_string(index=False))

print("\nsaved:", OUT)