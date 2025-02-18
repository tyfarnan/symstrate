import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_grid_examples(json_dir, output_dir='images', figsize=(12, 6)):
    """
    Generate and save side-by-side visualizations for all grid examples in a directory.
    
    Args:
        json_dir (str): Directory containing JSON files
        output_dir (str): Directory to save visualizations (defaults to 'images')
        figsize (tuple): Figure size for each visualization (width, height)
    """
    # Create output directory
    output_path = Path(output_dir)
    json_dir = Path(json_dir)
    
    # Create train and test subdirectories
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files in directory
    for json_path in json_dir.glob('*.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Process both train and test examples
        for dataset_type in ['train', 'test']:
            if dataset_type not in data:
                continue
                
            save_dir = test_dir if dataset_type == 'test' else train_dir
            examples = data[dataset_type]
            
            for idx, example in enumerate(examples):
                # Create figure with two subplots (input and output)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
                
                # Plot input grid
                input_grid = np.array(example['input'])
                im1 = ax1.imshow(input_grid, cmap='viridis')
                ax1.set_title('Input Grid')
                plt.colorbar(im1, ax=ax1)
                
                # Add grid lines
                for i in range(input_grid.shape[1]):
                    ax1.axvline(i - 0.5, color='black', linewidth=0.5)
                for j in range(input_grid.shape[0]):
                    ax1.axhline(j - 0.5, color='black', linewidth=0.5)
                
                # Plot output grid
                output_grid = np.array(example['output'])
                im2 = ax2.imshow(output_grid, cmap='viridis')
                ax2.set_title('Output Grid')
                plt.colorbar(im2, ax=ax2)
                
                # Add grid lines
                for i in range(output_grid.shape[1]):
                    ax2.axvline(i - 0.5, color='black', linewidth=0.5)
                for j in range(output_grid.shape[0]):
                    ax2.axhline(j - 0.5, color='black', linewidth=0.5)
                
                # Set main title with train/test indicator
                plt.suptitle(f'{dataset_type.capitalize()}: {json_path.stem} - Example {idx + 1}')
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(save_dir / f'{json_path.stem}_example_{idx + 1}.png', 
                           bbox_inches='tight', dpi=150)
                plt.close()
            
            print(f"Processed {len(examples)} {dataset_type} examples from {json_path.name}")

if __name__ == "__main__":
    visualize_grid_examples('training')
