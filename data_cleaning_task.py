#%%
import os
import pandas as pd

current_dir = os.getcwd()  # Works in notebooks!
file_path = os.path.join(current_dir, "Cleaned_Datasets", "task_0518.csv")
df = pd.read_csv(file_path)


# Identify BEGIN and END indices
begin_indices = df[df['Response'] == 'BEGIN'].index.tolist()
end_indices = df[df['Response'] == 'END'].index.tolist()

if len(begin_indices) != len(end_indices):
    raise ValueError("Mismatched BEGIN and END markers")

final_rows = []

for begin, end in zip(begin_indices, end_indices):
    block = df.iloc[begin:end+1]
    block = block[block['Response Type'] == 'response'].copy()

    if block.shape[0] != 16:
        continue

    participant_id = block['Participant Private ID'].iloc[0]
    task_name = block['Task Name'].iloc[0]
    display1 = block[block['Spreadsheet: display'] == 'Display 1']
    display2 = block[block['Spreadsheet: display'] == 'Display 2']

    def assign_product_group(display_block):
        hed_block = display_block.iloc[:4].copy()
        uti_block = display_block.iloc[4:8].copy()

        # Assign product name based on spreadsheet column
        hed_product = hed_block['Spreadsheet: hedonic'].iloc[0]
        uti_product = uti_block['Spreadsheet: uti'].iloc[0]

        hed_block['Product'] = hed_product
        uti_block['Product'] = uti_product

        return pd.concat([hed_block, uti_block], ignore_index=True)

    display1_cleaned = assign_product_group(display1)
    display2_cleaned = assign_product_group(display2)

    # Combine both displays
    full_block = pd.concat([display1_cleaned, display2_cleaned], ignore_index=True)
    full_block['Participant Private ID'] = participant_id
    full_block['Task Name'] = task_name

    final_rows.append(full_block)

# Combine all participants' responses
long_df = pd.concat(final_rows, ignore_index=True)
grouped = long_df.groupby(['Participant Private ID', 'Task Name', 'Product'])

wide_rows = []

for name, group in grouped:
    responses = group['Response'].tolist()
    row = {
        'Participant Private ID': name[0],
        'Task Name': name[1],
        'Product': name[2],
    }
    for i, r in enumerate(responses):
        row[f'Response_{i+1}'] = r
    wide_rows.append(row)

final_wide_df = pd.DataFrame(wide_rows)

# Save to CSV
output_path = os.path.join(current_dir, "product_responses.csv")
final_wide_df.to_csv(output_path, index=False)
print(f"Saved: {output_path}")
#%%
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "product_responses.csv"
file_path = os.path.join(current_dir, file_name)
df = pd.read_csv(file_path)

# Recode for purchase likelihood
likelihood_map = {
    "Very unlikely": 1,
    "Unlikely": 2,
    "Somewhat unlikely": 3,
    "Neutral": 4,
    "Somewhat likely": 5,
    "Likely": 6,
    "Very likely": 7
}

value_map = {
    "Not valuable at all": 1,
    "Slightly valuable": 2,
    "Somewhat valuable": 3,
    "Moderately valuable": 4,
    "Fairly valuable": 5,
    "Very valuable": 6,
    "Extremely valuable": 7
}

trust_map = {
    "Do not trust at all": 1,
    "Trust very little": 2,
    "Trust somewhat": 3,
    "Neutral": 4,
    "Trust fairly well": 5,
    "Trust a lot": 6,
    "Trust completely": 7
}

# Apply mappings to the appropriate columns
df['Response_2_recoded'] = df['Response_2'].map(likelihood_map)
df['Response_3_recoded'] = df['Response_3'].map(value_map)
df['Response_4_recoded'] = df['Response_4'].map(trust_map)

#%%
# Copy original
df_raw = df.copy()

# Rename for consistency
df_raw = df_raw.rename(columns={
    "Participant Private ID": "Participant_ID",
    "Task Name": "Task_Name",
    "Response_1": "Payment",
    "Response_2_recoded": "Intent",
    "Response_3_recoded": "Value",
    "Response_4_recoded": "Trust"
})
# Extract Condition from Task Name
df_raw["Condition"] = df_raw["Task_Name"].str.replace("_cue", "", regex=False).str.lower()
# Extract Context from Product
df_raw["Context"] = df_raw["Product"].apply(lambda x: "hedonic" if "hed" in x else "utilitarian")
# Extract Product from Product filename
df_raw["Product"] = df_raw["Product"].apply(lambda x: "chocolate" if "choc" in x else "headphones")

# Keep only relevant columns in the correct order
df_long = df_raw[["Participant_ID", "Condition", "Context", "Product", "Payment", "Intent", "Value", "Trust"]]

# Preview result
df_long.head()
df_long.to_csv("product_responses_recoded_cleaned.csv", index=False)

#%%
cleaned_dir = os.path.join(current_dir, "Cleaned_Datasets")
output_path = os.path.join(cleaned_dir, "task_cleaned.csv")
df_long.to_csv(output_path, index=False)

# %%
