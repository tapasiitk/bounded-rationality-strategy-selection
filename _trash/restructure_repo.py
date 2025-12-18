import os
import shutil
import glob

def restructure_repo():
    # 1. Define the new folder structure
    folders = ['data', 'scripts', 'notebooks', 'paper', '_trash']
    
    # 2. Define the specific file moves (Source Name -> New Path)
    # format: "current_filename": "new_folder/new_filename"
    file_map = {
        # Data
        "stimuli_multinum_25May25.csv": "data/stimuli.csv",
        "filtered_data_vpsyco_numerical.csv": "data/responses.csv",
        
        # Notebooks
        "analyse_multinum.ipynb": "notebooks/analysis_walkthrough.ipynb",
        
        # Scripts
        "model_fit_compare_code2.py": "scripts/01_model_analysis.py",
        "generate_fig.py": "scripts/02_plot_figures.py",
        
        # Paper (if present in root)
        "absdm2025_paper.pdf": "paper/absdm2025_paper.pdf"
    }

    # 3. Define files to move to trash (Exact matches)
    junk_files = [
        "model_fitting_complete_code.py",
        "model_validate.py",
        "pooled_data.csv",
        "restructure_repo.py" # Move self to trash after running
    ]

    print("--- Starting Repository Cleanup ---")

    # Create directories
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created directory: {folder}/")

    # Move and Rename important files
    for src, dst in file_map.items():
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved: {src} -> {dst}")
        else:
            print(f"⚠️  Skipped: {src} (File not found)")

    # Move junk files to _trash
    for junk in junk_files:
        if os.path.exists(junk):
            shutil.move(junk, os.path.join("_trash", junk))
            print(f"🗑️  Trashed: {junk}")

    # Move generated output files (wildcards) to _trash
    # (Matches any model_comparison csv or png files in the root)
    for wildcard in ["model_comparison_*.csv", "model_comparison_*.png", "*.png"]:
        for file_path in glob.glob(wildcard):
            # Don't move files that are already inside folders
            if os.path.isfile(file_path): 
                shutil.move(file_path, os.path.join("_trash", file_path))
                print(f"🗑️  Trashed output file: {file_path}")

    print("\n" + "="*50)
    print("DONE! The structure is now clean.")
    print("="*50)
    print("⚠️  ACTION REQUIRED ⚠️")
    print("1. Open 'scripts/01_model_analysis.py' and 'scripts/02_plot_figures.py'")
    print("2. Update the code to look for data in the 'data/' folder.")
    print("   Example: Change 'stimuli_multinum_25May25.csv' to 'data/stimuli.csv'")
    print("3. Check the '_trash' folder. If you are sure, delete it.")
    print("="*50)

if __name__ == "__main__":
    restructure_repo()
