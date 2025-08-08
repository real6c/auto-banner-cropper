#!/usr/bin/env python3
"""
Monochrome Analysis Script

This script analyzes a JPG image (converted to monochrome) to find the height of a region 
that has similar total blackness to the bottom row, then draws a red outline box around 
that region on the original image.

Usage: python color_analysis.py <input_image.jpg> [output_image.jpg] [--threshold 0.1]
"""

import sys
import numpy as np
from PIL import Image, ImageDraw
import argparse
from typing import Tuple, List





def analyze_image_color_region(image_path: str, threshold_percentage: float = 0.1, max_height: int = None) -> Tuple[int, Image.Image]:
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
    
    print(f"Initial average blackness (bottom row): {initial_avg:.2f}")
    
    # Check if bottom row has sufficient blackness for analysis
    if initial_avg < 100:  # Adjusted threshold for average blackness, default is 100
        print(f"Bottom row average blackness ({initial_avg:.2f}) is below minimum threshold (100). Skipping analysis.")
        return 0, image
    
    # Set default max_height to image height / 4 if not specified
    if max_height is None:
        max_height = height // 4
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
        
        print(f"Row {row_from_bottom} (from bottom): Average blackness {row_avg:.2f}, Percentage: {percentage:.3f}")
        
        # If percentage is less than threshold, stop
        if percentage < threshold_percentage:
            print(f"Stopping at row {row_from_bottom} (from bottom) - percentage {percentage:.3f} < {threshold_percentage}")
            break
        
        height_found += 1
        
        # If max_height is set and we've reached it, stop
        if height_found >= max_height:
            print(f"Stopping at row {row_from_bottom} (from bottom) - reached max height limit of {max_height} pixels")
            break
    
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


def main():
    parser = argparse.ArgumentParser(description='Analyze image blackness regions and draw bounding box')
    parser.add_argument('input_image', help='Input JPG image path')
    parser.add_argument('output_image', nargs='?', help='Output image path (optional)')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, 
                       help='Threshold percentage for stopping when dropped below (default: 0.5 = 50%%)')
    parser.add_argument('--max-height', '-m', type=int, default=None,
                       help='Maximum height in pixels to analyze (default: image height / 4)')
    
    args = parser.parse_args()
    
    try:
        # Analyze the image
        height_found, processed_image = analyze_image_color_region(args.input_image, args.threshold, args.max_height)
        
        # Determine output path
        if args.output_image:
            output_path = args.output_image
        else:
            # Generate default output name
            input_name = args.input_image.rsplit('.', 1)[0]
            output_path = f"{input_name}_analyzed.jpg"
        
        # Save the processed image
        processed_image.save(output_path)
        print(f"Processed image saved to: {output_path}")
        print(f"Region height: {height_found} pixels")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 