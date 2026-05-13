import os

ROOT = os.path.join(os.path.dirname(__file__), "outputs", "Mar-23-2026")

# Order matters: check the more specific substring first
REPLACEMENTS = [
    ("Meta-Llama-3.1-70B-Instruct-bnb-4bit", "Llama-3.1-70B-Instruct"),
    ("Meta-Llama-3.1-70B-bnb-4bit", "Llama-3.1-70B"),
]

renamed = 0
for dirpath, _, filenames in os.walk(ROOT):
    for filename in filenames:
        if not filename.endswith(".csv"):
            continue
        new_name = filename
        for old, new in REPLACEMENTS:
            if old in new_name:
                new_name = new_name.replace(old, new)
                break
        if new_name != filename:
            src = os.path.join(dirpath, filename)
            dst = os.path.join(dirpath, new_name)
            os.rename(src, dst)
            print(f"  {filename}  ->  {new_name}")
            renamed += 1

print(f"\nDone. {renamed} file(s) renamed.")
