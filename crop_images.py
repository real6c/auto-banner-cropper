#!/usr/bin/env python3
"""
Image Cropping Script

This script analyzes JPG images using the same logic as color_analysis.py and crops
the bottom portion based on the detected height. It processes entire datasets while
preserving the input directory structure.

Usage: python crop_images.py <input_directory> <output_directory> [--threshold 0.5] [--max-height 100]
"""

import sys
import os
import numpy as np
from PIL import Image
import argparse
from typing import Tuple, List
from pathlib import Path
from PIL import ImageDraw
from tqdm import tqdm


def analyze_image_color_region(image_path: str, threshold_percentage: float = 0.1, max_height: int = None, verbose: bool = False) -> Tuple[int, Image.Image]:
    """
    Analyze image to find height of region with similar average blackness to bottom row.
    
    Blackness is calculated as (255 - grayscale_value) so that:
    - Darker pixels = higher blackness values
    - Lighter pixels = lower blackness values
    
    Uses average blackness per pixel to normalize for different image widths.
    If bottom row average blackness is below 100, returns original image without analysis.
    
    Args:
        image_path: Path to input image
        threshold_percentage: Percentage threshold for stopping when dropped below (default 0.1 = 10%)
        max_height: Maximum height in pixels to analyze (default None = image height / 4)
        verbose: If True, print detailed analysis output
    
    Returns:
        Tuple of (height, processed_image)
    """
    # Open and convert image to RGB for output
    image = Image.open(image_path).convert('RGB')
    
    # Create monochrome version for analysis
    monochrome_image = Image.open(image_path).convert('L')  # Convert to grayscale
    mono_array = np.array(monochrome_image)
    
    height, width = mono_array.shape
    
    # Get the average blackness of the bottom row (sum of all pixels / number of pixels)
    # In grayscale, 0 = black, 255 = white, so we invert to get proper blackness
    # Blackness = 255 - grayscale_value (so 0 becomes 255, 255 becomes 0)
    initial_avg = np.mean(255 - mono_array[height-1, :])
    
    if verbose:
        print(f"Initial average blackness (bottom row): {initial_avg:.2f}")
    
    # Check if bottom row has sufficient blackness for analysis
    if initial_avg < 100:  # Adjusted threshold for average blackness, default is 100
        if verbose:
            print(f"Bottom row average blackness ({initial_avg:.2f}) is below minimum threshold (100). Skipping analysis.")
        return 0, image
    
    # Set default max_height to image height / 4 if not specified
    if max_height is None:
        max_height = height // 4
        if verbose:
            print(f"Using default max height: {max_height} pixels (image height / 4)")
    
    # Iterate from bottom to top
    height_found = 0
    for row in range(height-1, -1, -1):  # Start from bottom (height-1) to top (0)
        # Get average blackness of current row (sum of all pixels / number of pixels)
        # Invert grayscale values: 255 - value gives us proper blackness
        row_avg = np.mean(255 - mono_array[row, :])
        
        # Calculate percentage of initial value (how many times the current blackness is compared to initial)
        percentage = row_avg / initial_avg
        
        # Calculate row number from bottom (0 = bottom row)
        row_from_bottom = height - 1 - row
        
        if verbose:
            print(f"Row {row_from_bottom} (from bottom): Average blackness {row_avg:.2f}, Percentage: {percentage:.3f}")
        
        # If percentage is less than threshold, stop
        if percentage < threshold_percentage:
            if verbose:
                print(f"Stopping at row {row_from_bottom} (from bottom) - percentage {percentage:.3f} < {threshold_percentage}")
            break
        
        height_found += 1
        
        # If max_height is set and we've reached it, stop
        if height_found >= max_height:
            if verbose:
                print(f"Stopping at row {row_from_bottom} (from bottom) - reached max height limit of {max_height} pixels")
            break
    
    if verbose:
        print(f"Height of similar blackness region: {height_found} pixels")
    
    # Draw red outline box on original RGB image
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Calculate box coordinates
    # Box goes from bottom up to the height we found
    box_top = height - height_found
    box_bottom = height - 1
    box_left = 0
    box_right = width - 1
    
    # Draw rectangle outline (2 pixel width for visibility)
    draw.rectangle(
        [box_left, box_top, box_right, box_bottom],
        outline='red',
        width=2
    )
    
    return height_found, draw_image


def crop_image_by_height(image_path: str, crop_height: int) -> Image.Image:
    """
    Crop an image by removing the bottom portion based on the detected height.
    
    Args:
        image_path: Path to input image
        crop_height: Height in pixels to crop from bottom
    
    Returns:
        Cropped image
    """
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    if crop_height <= 0:
        return image
    
    # Crop from top to (height - crop_height)
    cropped_image = image.crop((0, 0, width, height - crop_height))
    return cropped_image


def create_comparison_image(original_path: str, cropped_image: Image.Image, crop_height: int) -> Image.Image:
    """
    Create a side-by-side comparison of original and cropped images.
    
    Args:
        original_path: Path to original image
        cropped_image: Cropped image
        crop_height: Height that was cropped
    
    Returns:
        Side-by-side comparison image
    """
    original = Image.open(original_path).convert('RGB')
    
    # Get dimensions
    orig_width, orig_height = original.size
    crop_width, crop_height_actual = cropped_image.size
    
    # Create a new image with side-by-side layout
    # Width: original width + cropped width + padding
    # Height: max of original height and cropped height
    padding = 20
    total_width = orig_width + crop_width + padding
    total_height = max(orig_height, crop_height_actual)
    
    comparison = Image.new('RGB', (total_width, total_height), color='white')
    
    # Paste original image on the left
    comparison.paste(original, (0, 0))
    
    # Paste cropped image on the right
    comparison.paste(cropped_image, (orig_width + padding, 0))
    
    # Add text labels
    draw = ImageDraw.Draw(comparison)
    
    # Try to use a default font, fallback to basic if not available
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add labels
    draw.text((10, 10), f"Original ({orig_width}x{orig_height})", fill='black', font=font)
    draw.text((orig_width + padding + 10, 10), f"Cropped ({crop_width}x{crop_height_actual})", fill='black', font=font)
    
    if crop_height > 0:
        draw.text((orig_width + padding + 10, 30), f"Cropped {crop_height}px from bottom", fill='red', font=font)
    else:
        draw.text((orig_width + padding + 10, 30), "No cropping needed", fill='green', font=font)
    
    return comparison


def process_single_image(input_path: str, output_path: str, threshold: float, max_height: int = None, create_comparison: bool = False, verbose: bool = False) -> bool:
    """
    Process a single image: analyze and crop if necessary.
    
    Args:
        input_path: Path to input image
        output_path: Path to output image
        threshold: Threshold percentage for analysis
        max_height: Maximum height to analyze
        create_comparison: If True, create side-by-side comparison instead of just cropped image
        verbose: If True, print detailed output for each image
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if verbose:
            print(f"\nProcessing: {input_path}")
        
        # Analyze the image to find crop height
        crop_height, _ = analyze_image_color_region(input_path, threshold, max_height, verbose)
        
        if crop_height > 0:
            # Crop the image
            cropped_image = crop_image_by_height(input_path, crop_height)
            if verbose:
                print(f"Cropping {crop_height} pixels from bottom")
        else:
            # No cropping needed, use original image
            cropped_image = Image.open(input_path).convert('RGB')
            if verbose:
                print("No cropping needed")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the appropriate image
        if create_comparison:
            # Create side-by-side comparison
            comparison_image = create_comparison_image(input_path, cropped_image, crop_height)
            comparison_image.save(output_path)
            if verbose:
                print(f"Saved comparison to: {output_path}")
        else:
            # Save just the cropped image
            cropped_image.save(output_path)
            if verbose:
                print(f"Saved cropped image to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def find_image_files(directory: str) -> List[str]:
    """
    Recursively find all image files in a directory.
    
    Args:
        directory: Root directory to search
    
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files


def main():
    parser = argparse.ArgumentParser(description='Crop images based on color analysis')
    parser.add_argument('input_directory', help='Input directory containing images')
    parser.add_argument('output_directory', help='Output directory for processed images')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                       help='Threshold percentage for stopping when dropped below (default: 0.5 = 50%%)')
    parser.add_argument('--max-height', '-m', type=int, default=None,
                       help='Maximum height in pixels to analyze (default: image height / 4)')
    parser.add_argument('--comparison', '-c', action='store_true',
                       help='Create side-by-side before/after comparison images instead of just cropped images')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed output for each image (disables progress bar)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_directory, exist_ok=True)
    
    # Find all image files
    print(f"Scanning for images in: {args.input_directory}")
    image_files = find_image_files(args.input_directory)
    print(f"Found {len(image_files)} image files")
    
    if not image_files:
        print("No image files found in the input directory")
        sys.exit(1)
    
    # Process each image with progress bar or verbose output
    successful = 0
    failed = 0
    
    if args.verbose:
        print(f"\nProcessing {len(image_files)} images with detailed output...")
        for image_path in image_files:
            # Calculate relative path to preserve directory structure
            rel_path = os.path.relpath(image_path, args.input_directory)
            output_path = os.path.join(args.output_directory, rel_path)
            
            if process_single_image(image_path, output_path, args.threshold, args.max_height, args.comparison, verbose=True):
                successful += 1
            else:
                failed += 1
    else:
        print(f"\nProcessing {len(image_files)} images...")
        for image_path in tqdm(image_files, desc="Processing images", unit="image"):
            # Calculate relative path to preserve directory structure
            rel_path = os.path.relpath(image_path, args.input_directory)
            output_path = os.path.join(args.output_directory, rel_path)
            
            if process_single_image(image_path, output_path, args.threshold, args.max_height, args.comparison, verbose=False):
                successful += 1
            else:
                failed += 1
    
    # Summary
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")


if __name__ == "__main__":
    main() 