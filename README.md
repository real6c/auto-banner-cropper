# Auto Banner Cropper

Automatically crops banner/watermark regions from images using intelligent color analysis, great for large webscraped image datasets.

## Setup

1. **Create Virtual Environment**:
```bash
python -m venv autocrop-venv
source autocrop-venv/bin/activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

## Scripts

### `color_analysis.py` - Single Image Analysis
Analyzes a single image and draws a red outline around detected banner regions.

**Usage**:
```bash
python color_analysis.py input_image.jpg [output_image.jpg] [--threshold 0.1]
```

**Parameters**:
- `input_image.jpg`: Image to analyze
- `output_image.jpg`: Optional output file (default: adds "_analyzed" suffix)
- `--threshold`: Percentage threshold for stopping analysis (default: 0.1 = 10%)

**What it does**:
- Converts image to grayscale and analyzes blackness per row
- Detects regions with similar blackness to the bottom row
- Draws a red outline box around the detected banner region
- Shows detailed analysis output

### `crop_images.py` - Batch Image Cropping
Processes entire directories of images, cropping banner regions while preserving folder structure.

**Usage**:
```bash
python crop_images.py input_directory output_directory [options]
```

**Parameters**:
- `input_directory`: Directory containing images to process
- `output_directory`: Directory where cropped images will be saved
- `--threshold, -t`: Threshold percentage (default: 0.5 = 50%)
- `--max-height, -m`: Maximum height to analyze in pixels (default: image height / 4)
- `--comparison, -c`: Create side-by-side before/after comparison images
- `--verbose, -v`: Show detailed output for each image

**What it does**:
- Recursively processes all images in input directory
- Preserves folder structure in output directory
- Crops detected banner regions from bottom of images
- Shows progress bar for batch processing
- Optionally creates comparison images showing before/after

### `sample_images.py` - Random Image Sampling
Randomly samples a specified number of images from a directory.

**Usage**:
```bash
python sample_images.py input_directory output_directory num_samples [--preserve-structure] [--seed SEED]
```

**Parameters**:
- `input_directory`: Directory containing images to sample from
- `output_directory`: Directory where sampled images will be copied
- `num_samples`: Number of images to randomly sample
- `--preserve-structure, -p`: Preserve directory structure in output (default: flatten structure)
- `--seed, -s`: Random seed for reproducible sampling (optional)

**What it does**:
- Recursively finds all images in input directory
- Randomly selects specified number of images
- Copies selected images to output directory (flattened by default)
- Optionally preserves folder structure when using `--preserve-structure`
- Shows progress bar during copying
- Provides summary of successful/failed operations

## Example Usage

```bash
# Analyze single image
python color_analysis.py photo.jpg --threshold 0.3

# Crop all images in a directory
python crop_images.py ./input_images ./output_images --comparison

# Crop with custom settings
python crop_images.py ./input_images ./output_images --threshold 0.4 --max-height 200

# Sample 100 random images (flattened structure)
python sample_images.py ./dataset ./sample 100

# Sample with preserved directory structure
python sample_images.py ./dataset ./sample 100 --preserve-structure

# Sample with reproducible random seed
python sample_images.py ./dataset ./sample 50 --seed 42
```

## Supported Formats
JPG, PNG, BMP, TIFF 
