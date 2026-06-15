import glob
import os

def update_kmc_headers():
    # Find all files matching the naming convention in the current directory
    log_files = glob.glob("benchmark_*K_seed*.log")
    
    if not log_files:
        print("No log files found matching the pattern 'benchmark_*K_seed*.log'.")
        return

    # Dictionary mapping the old header labels to the new v0.6.0 labels
    replacements = {
        "### KMC Simulator v0.5.2 #############": "### KMC Simulator v0.6.0 #############",
        "# Constant:": "# Constants:",
        "# Initial Vacancies:": "# Initial Vacancies:",
        "# Barriers Source File:": "# Barriers Source File:",
        "# Log Format:": "# Log Format:"
    }

    count = 0
    for file_path in log_files:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        in_header = True

        # Process only the header lines to save time and prevent accidental edits below
        for i in range(len(lines)):
            if in_header:
                # Apply the label replacements
                for old_str, new_str in replacements.items():
                    if old_str in lines[i]:
                        lines[i] = lines[i].replace(old_str, new_str)
                
                # Stop parsing once we hit the bottom boundary of the header
                if "######################################" in lines[i] and i > 0:
                    in_header = False
            else:
                break 

        # Overwrite the file with the newly updated lines
        with open(file_path, 'w') as file:
            file.writelines(lines)
            
        count += 1
        print(f"Updated: {file_path}")

    print(f"\nSuccessfully updated headers for {count} files.")

if __name__ == "__main__":
    update_kmc_headers()
