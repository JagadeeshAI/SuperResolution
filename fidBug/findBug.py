import os
import numpy as np
import pandas as pd

# Directory paths
unet_dir = "/media/jagadeesh/New Volume/Jagadeesh/SuperResolution/fidBug/unet_17_44/Restomer_results"
restomer_dir = "/media/jagadeesh/New Volume/Jagadeesh/SuperResolution/fidBug/Restomer_results_22/Restomer_results"

# Output text file path
output_file = "/media/jagadeesh/New Volume/Jagadeesh/SuperResolution/fidBug/comparison_results.txt"

# Function to extract info from .npz files in a directory
def extract_npz_info(directory):
    results = {}
    
    # Get list of all .npz files in the directory
    npz_files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    
    for file_name in npz_files:
        file_path = os.path.join(directory, file_name)
        try:
            # Load the .npz file
            data = np.load(file_path)
            
            # Extract file number without extension for matching
            file_num = os.path.splitext(file_name)[0]
            
            results[file_num] = {}
            
            # Process each array in the file
            for arr_name in data.files:
                arr = data[arr_name]
                
                # Store shape and max value
                if arr_name == 'raw':
                    results[file_num]['shape'] = arr.shape
                    results[file_num]['max_raw'] = np.max(arr)
                elif arr_name == 'max_val':
                    results[file_num]['max_val'] = float(arr)
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    
    return results

# Extract info from both directories
print("Processing UNet results...")
unet_results = extract_npz_info(unet_dir)
print("Processing Restomer results...")
restomer_results = extract_npz_info(restomer_dir)

# Create a combined set of all file numbers
all_files = sorted(list(set(unet_results.keys()).union(set(restomer_results.keys()))), key=lambda x: int(x) if x.isdigit() else x)

# Create DataFrame for tabular output
comparison_data = []

for file_num in all_files:
    row = {"File": file_num}
    
    # Add data from UNet directory if available
    if file_num in unet_results:
        row["UNet_Shape"] = str(unet_results[file_num].get('shape', 'N/A'))
        row["UNet_Max_Raw"] = unet_results[file_num].get('max_raw', 'N/A')
        row["UNet_Max_Val"] = unet_results[file_num].get('max_val', 'N/A')
    else:
        row["UNet_Shape"] = "N/A"
        row["UNet_Max_Raw"] = "N/A"
        row["UNet_Max_Val"] = "N/A"
    
    # Add data from Restomer directory if available
    if file_num in restomer_results:
        row["Restomer_Shape"] = str(restomer_results[file_num].get('shape', 'N/A'))
        row["Restomer_Max_Raw"] = restomer_results[file_num].get('max_raw', 'N/A')
        row["Restomer_Max_Val"] = restomer_results[file_num].get('max_val', 'N/A')
    else:
        row["Restomer_Shape"] = "N/A"
        row["Restomer_Max_Raw"] = "N/A"
        row["Restomer_Max_Val"] = "N/A"
    
    comparison_data.append(row)

# Create DataFrame
df = pd.DataFrame(comparison_data)

# Set up multi-level columns for better organization
df.columns = pd.MultiIndex.from_tuples([
    ('File', ''),
    ('UNet Results', 'Shape'),
    ('UNet Results', 'Max Raw'),
    ('UNet Results', 'Max Val'),
    ('Restomer Results', 'Shape'),
    ('Restomer Results', 'Max Raw'),
    ('Restomer Results', 'Max Val')
])

# Generate text table and save to file
with open(output_file, "w") as f:
    f.write("Comparison of UNet and Restomer Results\n")
    f.write("=====================================\n\n")
    f.write(f"UNet Directory: {unet_dir}\n")
    f.write(f"Restomer Directory: {restomer_dir}\n\n")
    
    # Write table to file with proper formatting
    f.write(df.to_string(index=False))
    
    # Add ratio analysis
    f.write("\n\n")
    f.write("Raw Value Ratio Analysis (UNet/Restomer):\n")
    f.write("=================================\n")
    
    for file_num in all_files:
        if file_num in unet_results and file_num in restomer_results:
            if 'max_raw' in unet_results[file_num] and 'max_raw' in restomer_results[file_num]:
                if restomer_results[file_num]['max_raw'] != 0:
                    ratio = unet_results[file_num]['max_raw'] / restomer_results[file_num]['max_raw']
                    f.write(f"File {file_num}: {ratio:.4f}\n")
                else:
                    f.write(f"File {file_num}: Infinity (division by zero)\n")

print(f"Comparison complete. Results saved to {output_file}")

