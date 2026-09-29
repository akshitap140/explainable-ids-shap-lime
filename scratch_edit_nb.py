import json
import os

with open('/Users/bsharishkumar/explainable-ids-shap-lime/notebooks/data_preprocssing_cicds.ipynb', 'r') as f:
    nb = json.load(f)

# 1. Add os.chdir('..')
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'import pandas' in ''.join(cell['source']):
        source = cell['source']
        for i, line in enumerate(source):
            if 'import os' in line:
                source.insert(i+1, "if os.path.basename(os.getcwd()) == 'notebooks':\n")
                source.insert(i+2, "    os.chdir('..')\n")
                break
        break

# 2. Add Feature Reduction section
reduction_markdown = {
    "cell_type": "markdown",
    "id": "feature-reduction-section",
    "metadata": {},
    "source": [
        "## 6. Feature Reduction\n",
        "Select the most important features to reduce dimensionality."
    ]
}

reduction_code = {
    "cell_type": "code",
    "execution_count": None,
    "id": "feature_reduction",
    "metadata": {},
    "outputs": [],
    "source": [
        "# Using the RF importances calculated earlier to select top 20 features\n",
        "top_features = importances.head(20).index.tolist()\n",
        "print(f'Selecting Top 20 features: {top_features}')\n",
        "\n",
        "# Keep metadata/label columns as well\n",
        "label_cols = [LABEL_COL, 'Label_binary', 'Label_encoded', 'Label_multiclass']\n",
        "cols_to_keep = top_features + [c for c in label_cols if c in df.columns]\n",
        "\n",
        "df_reduced = df[cols_to_keep].copy()\n",
        "print(f'Shape after feature reduction: {df_reduced.shape}')\n"
    ]
}

# 3. Modify Save cell to save the reduced data
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'OUTPUT_PATH' in ''.join(cell['source']):
        source = cell['source']
        for i, line in enumerate(source):
            if 'df.to_csv' in line:
                source[i] = "df_reduced.to_csv(OUTPUT_PATH, index=False)\n"
            if 'OUTPUT_PATH =' in line:
                source[i] = "OUTPUT_PATH = 'data/preprocessed/raw/CICDS_reduced.csv'\n"
            if 'feature_cols =' in line:
                source[i] = "feature_cols = [c for c in df_reduced.columns if c not in [LABEL_COL, 'Label_binary', 'Label_encoded', 'Label_multiclass']]\n"
            if 'df[feature_cols].isna()' in line:
                source[i] = "assert not df_reduced[feature_cols].isna().any().any(), 'NaNs remain in features'\n"
            if 'numeric_df_cols = df.' in line:
                source[i] = "numeric_df_cols = df_reduced.select_dtypes(include=[np.number]).columns\n"
            if 'df[numeric_df_cols]' in line:
                source[i] = "assert not np.isinf(df_reduced[numeric_df_cols]).any().any(), 'Infs remain in features'\n"
            if 'df.shape' in line:
                source[i] = "print(f'Final shape: {df_reduced.shape}')\n"
            if 'len(df.columns)' in line:
                source[i] = "print(f'Columns: {len(df_reduced.columns)}')\n"
            if 'df.isna().sum()' in line:
                source[i] = "print(f'Remaining NaNs: {int(df_reduced.isna().sum().sum())}')\n"
            if 'df[' in line and 'value_counts' in line:
                # To handle df['Label_binary'].value_counts() etc.
                source[i] = source[i].replace('df[', 'df_reduced[')
                
        # Insert reduction cells before the Save section (which usually has a markdown header before it)
        break

# Find where to insert reduction cells (before "Save Cleaned Data" markdown)
save_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Save Cleaned Data' in ''.join(cell['source']):
        save_idx = i
        break

if save_idx != -1:
    nb['cells'].insert(save_idx, reduction_code)
    nb['cells'].insert(save_idx, reduction_markdown)
    # Also update the markdown title for saving
    nb['cells'][save_idx + 2]['source'][0] = "## 7. Save Reduced Data"

with open('/Users/bsharishkumar/explainable-ids-shap-lime/notebooks/data_preprocssing_cicds.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
