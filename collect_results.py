import os
import shutil
import argparse

def collect_results(directory):
    """Move result.pkl files into their corresponding figure folders"""
    # Get all result.pkl files
    result_files = [f for f in os.listdir(directory) 
                   if f.endswith('.result.pkl')]
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            # Get the base name (remove .result.pkl)
            base_name = result_file[:-11]  # removes '.result.pkl'
            
            # Construct figure folder path
            figure_folder = os.path.join(directory, f"{base_name}_figures")
            
            if os.path.exists(figure_folder):
                # Copy the result file to the figure folder
                src = os.path.join(directory, result_file)
                dst = os.path.join(figure_folder, result_file)
                shutil.copy2(src, dst)
                print(f"Copied {result_file} to {figure_folder}")
            else:
                print(f"Warning: Figure folder not found for {base_name}")
                
        except Exception as e:
            print(f"Error processing {result_file}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Collect result.pkl files into figure folders")
    parser.add_argument("directory", help="Directory containing .result.pkl files and figure folders")
    
    args = parser.parse_args()
    collect_results(args.directory) 