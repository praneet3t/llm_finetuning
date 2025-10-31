import re
import pandas as pd
import json

# Read only first 20 rows, strip any surrounding quotes from headers
df = pd.read_csv(
    "/home/incois/tvsubhaskar/llm_project/TCPD_QH.tsv",
    sep='\t', quotechar='"', nrows=20
)
# Clean header names if they include extra quotes or spaces
df.columns = df.columns.str.strip().str.strip('"')

# Helper: split multi-part text into [(label, content)]
def split_parts(text):
    parts = re.split(r"\(([a-z])\)", str(text))
    it = iter(parts[1:])
    return [(lbl, cnt.strip()) for lbl, cnt in zip(it, it)]

# Expand one DataFrame row into multiple Q&A records
def expand_row(row):
    q_parts = split_parts(row['question_text'])
    a_parts = split_parts(row['answer_text'])
    answer_dict = {lbl: txt for lbl, txt in a_parts}

    records = []
    for qlbl, qtxt in q_parts:
        ans_txt = answer_dict.get(qlbl, row['answer_text'])
        rec = {
            'id': row['id'],
            'date': row['date'],
            'ls_number': row.get('ls_number', ''),
            'ministry': row.get('ministry', ''),
            'question_type': row.get('question_type', ''),
            'member': row.get('member', ''),
            'party': row.get('party', ''),
            'state': row.get('state', ''),
            'constituency': row.get('constituency', ''),
            'constituency_type': row.get('constituency_type', ''),
            'gender': row.get('gender', ''),
            'subject': row.get('subject', ''),
            'link': row.get('link', ''),
            'qa_label': qlbl,
            'question': qtxt,
            'answer': ans_txt
        }
        records.append(rec)
    return records

# Process first 20 rows
total = []
for _, r in df.iterrows():
    total.extend(expand_row(r))

# Write to JSONL
with open('first20.jsonl', 'w', encoding='utf-8') as fout:
    for rec in total:
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')

print(f"Extracted {len(total)} Q&A pairs into first20.jsonl")
