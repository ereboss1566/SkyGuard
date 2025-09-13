import pandas as pd
import os

# Define the directory containing the Excel files
excel_dir = 'data/historical'
csv_dir = 'data/historical/csv'

# Create the CSV directory if it doesn't exist
os.makedirs(csv_dir, exist_ok=True)

# List of Excel files to convert
excel_files = [
    'Air quality information.xlsx',
    'Astronomical.xlsx',
    'Location information.xlsx',
    'Weather data.xlsx'
]

# Convert each Excel file to CSV
for excel_file in excel_files:
    # Read the Excel file
    excel_path = os.path.join(excel_dir, excel_file)
    
    # Handle different sheet structures
    try:
        # Read all sheets
        excel_data = pd.read_excel(excel_path, sheet_name=None)
        
        # If there are multiple sheets, save each one as a separate CSV
        if len(excel_data) > 1:
            for sheet_name, df in excel_data.items():
                # Clean the sheet name for use in filename
                clean_sheet_name = sheet_name.replace(' ', '_').replace('/', '_')
                csv_filename = f"{os.path.splitext(excel_file)[0]}_{clean_sheet_name}.csv"
                csv_path = os.path.join(csv_dir, csv_filename)
                df.to_csv(csv_path, index=False)
                print(f"Converted {excel_file} sheet '{sheet_name}' to {csv_path}")
        else:
            # If there's only one sheet, save it directly
            df = list(excel_data.values())[0]
            csv_filename = f"{os.path.splitext(excel_file)[0]}.csv"
            csv_path = os.path.join(csv_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            print(f"Converted {excel_file} to {csv_path}")
    except Exception as e:
        print(f"Error converting {excel_file}: {str(e)}")

print("All Excel files have been converted to CSV format.")