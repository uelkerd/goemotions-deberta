import json

# Load the notebook
with open('notebooks/GoEmotions_DeBERTa_IMPROVED.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the cell with the monitor_processes() call
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Check if the cell contains the problematic call
        if any('monitor_processes()' in line for line in source):
            # Find the lines to remove
            lines_to_remove = [
                "\n",
                "# Call monitoring to demonstrate live monitoring\n",
                "monitor_processes()"
            ]
            # Remove the lines from the end
            if source[-3:] == lines_to_remove:
                cell['source'] = source[:-3]
                print("Removed monitor_processes() call from PHASE 4 cell")
                break

# Save the notebook
with open('notebooks/GoEmotions_DeBERTa_IMPROVED.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully")