import json

nb_path = '/Users/bsharishkumar/explainable-ids-shap-lime/notebooks/data_preprocssing_cicds.ipynb'
with open(nb_path, 'r') as f:
    nb = json.load(f)

more_fe_code = [
    "\n",
    "# Additional Feature Engineering\n",
    "df['fwd_packet_length_range'] = df['Fwd Packet Length Max'] - df['Fwd Packet Length Min']\n",
    "df['bwd_packet_length_range'] = df['Bwd Packet Length Max'] - df['Bwd Packet Length Min']\n",
    "df['flow_duration_log'] = np.log1p(df['Flow Duration'])\n",
    "df['total_fwd_bytes_log'] = np.log1p(df['Total Length of Fwd Packets'])\n",
    "df['total_bwd_bytes_log'] = np.log1p(df['Total Length of Bwd Packets'])\n",
    "df['byte_to_packet_ratio'] = df['Flow Bytes/s'] / (df['Flow Packets/s'] + 1)\n",
    "df['fwd_header_to_pkt_ratio'] = df['Fwd Header Length'] / (df['Total Fwd Packets'] + 1)\n",
    "df['bwd_header_to_pkt_ratio'] = df['Bwd Header Length'] / (df['Total Backward Packets'] + 1)\n"
]

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'df[\'total_flags\'] =' in ''.join(cell['source']):
        source = cell['source']
        
        # find where '# New feature count' starts
        insert_idx = -1
        for i, line in enumerate(source):
            if '# New feature count' in line:
                insert_idx = i
                break
        
        if insert_idx != -1:
            # insert more_fe_code before # New feature count
            for line in reversed(more_fe_code):
                source.insert(insert_idx, line)
        break

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)
