#!/usr/bin/env python3
"""
Random Image Sampling Script

This script randomly samples a specified number of images from an input directory
and copies them to an output directory while preserving the folder structure.

Usage: python sample_images.py <input_directory> <output_directory> <num_samples>
"""

import sys
import os
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def find_image_files(directory: str) -> List[str]:
    """
    Recursively find all image files in the given directory.
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of full paths to image files
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files


def get_relative_path(full_path: str, base_directory: str) -> str:
    """
    Get the relative path from the base directory.
    
    Args:
        full_path: Full path to the file
        base_directory: Base directory to calculate relative path from
        
    Returns:
        Relative path string
    """
    return os.path.relpath(full_path, base_directory)


def sample_and_copy_images(input_dir: str, output_dir: str, num_samples: int, preserve_structure: bool = False, seed: int = None) -> None:
    """
    Randomly sample images from input directory and copy them to output directory.
    
    Args:
        input_dir: Input directory containing images
        output_dir: Output directory where sampled images will be copied
        num_samples: Number of images to sample
        preserve_structure: Whether to preserve directory structure (default: False)
        seed: Random seed for reproducible sampling (optional)
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        print(f"Using random seed: {seed}")
    
    # Find all image files
    print(f"Scanning for images in: {input_dir}")
    all_images = find_image_files(input_dir)
    
    if not all_images:
        print("No image files found in the input directory.")
        return
    
    print(f"Found {len(all_images)} image files")
    
    # Check if we have enough images
    if num_samples > len(all_images):
        print(f"Warning: Requested {num_samples} samples but only {len(all_images)} images found.")
        print(f"Will sample all {len(all_images)} images.")
        num_samples = len(all_images)
    
    # Randomly sample images
    sampled_images = random.sample(all_images, num_samples)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy sampled images
    print(f"Copying {len(sampled_images)} images to: {output_dir}")
    if preserve_structure:
        print("Preserving directory structure")
    else:
        print("Flattening directory structure")
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(tqdm(sampled_images, desc="Copying images")):
        try:
            if preserve_structure:
                # Get relative path from input directory
                relative_path = get_relative_path(image_path, input_dir)
                
                # Create output path
                output_path = os.path.join(output_dir, relative_path)
                
                # Create output directory structure if it doesn't exist
                output_file_dir = os.path.dirname(output_path)
                os.makedirs(output_file_dir, exist_ok=True)
            else:
                # Flatten structure - just use filename with counter to avoid conflicts
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_{i:04d}{ext}"
                output_path = os.path.join(output_dir, output_filename)
            
            # Copy the file
            shutil.copy2(image_path, output_path)
            successful += 1
            
        except Exception as e:
            print(f"Failed to copy {image_path}: {e}")
            failed += 1
    
    # Print summary
    print(f"\nSampling complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(sampled_images)}")
    print(f"Output directory: {output_dir}")


def main():
    """Main function to handle command line arguments and execute sampling."""
    parser = argparse.ArgumentParser(
        description="Randomly sample images from a directory while preserving folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sample_images.py ./dataset ./sample 100
  python sample_images.py ./images ./test_sample 50 --seed 42
  python sample_images.py ./dataset ./sample 100 --preserve-structure
        """
    )
    
    parser.add_argument(
        "input_directory",
        help="Directory containing images to sample from"
    )
    
    parser.add_argument(
        "output_directory", 
        help="Directory where sampled images will be copied"
    )
    
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of images to randomly sample"
    )
    
    parser.add_argument(
        "--preserve-structure", "-p",
        action="store_true",
        help="Preserve directory structure in output (default: flatten structure)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' does not exist.")
        sys.exit(1)
    
    # Validate number of samples
    if args.num_samples <= 0:
        print("Error: Number of samples must be positive.")
        sys.exit(1)
    
    # Execute sampling
    try:
        sample_and_copy_images(
            args.input_directory,
            args.output_directory,
            args.num_samples,
            args.preserve_structure,
            args.seed
        )
    except KeyboardInterrupt:
        print("\nSampling interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during sampling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 