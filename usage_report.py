import os
import pandas as pd

# --- CONFIG ---

REPO_ROOT = r"C:\junk\MLMS" 
XLSX_INPUT = r"C:\Users\jay.parashar\Downloads\hive_schema_table_list_mine.xlsx"  
XLSX_OUTPUT = r"C:\junk\MLMS_table_usage_report.xlsx"
FILE_EXTENSIONS = {".py", ".sql", ".scala", ".ipynb", ".txt"}  # Add more if needed
# ---------------


def find_usage_in_file(filepath, search_token):
    """Return True if token is found in the file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if search_token in line:
                    return True
    except:
        pass
    return False


def main():
    # Load Excel input
    df = pd.read_excel(XLSX_INPUT)

    # Ensure expected columns exist
    for col in ["schema", "table", "Actively_Used", "File Name"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from Excel file!")

    # Build list of repo files to scan
    repo_files = []
    for root, _, files in os.walk(REPO_ROOT):
        for f in files:
            if any(f.endswith(ext) for ext in FILE_EXTENSIONS):
                repo_files.append(os.path.join(root, f))

    # Process each schema.table
    updated_rows = []

    for _, row in df.iterrows():
        schema = str(row["schema"])
        table = str(row["table"])
        token = f"{schema}.{table}"

        used_files = []

        for filepath in repo_files:
            if find_usage_in_file(filepath, token):
                used_files.append(filepath)

        # Update columns
        actively_used = "Y" if used_files else "N"
        file_list = ", ".join(used_files)

        updated_rows.append({
            "schema": schema,
            "table": table,
            "Actively_Used": actively_used,
            "File Name": file_list
        })

    # Save result
    out_df = pd.DataFrame(updated_rows)
    out_df.to_excel(XLSX_OUTPUT, index=False)

    print(f"Usage report created: {XLSX_OUTPUT}")


if __name__ == "__main__":
    main()
