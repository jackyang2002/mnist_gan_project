#!/usr/bin/env python3
"""
Create a 5x4 training progress grid showing generation quality improvement.
This script combines sample images from different epochs into a single grid.
"""

import os
from PIL import Image
import numpy as np

def create_training_progress_grid():
    """Create a 5x4 grid showing training progress across epochs."""
    
    # Define the epochs to show (5 rows, 4 columns = 20 epochs)
    epochs_to_show = [
        [1, 5, 10, 15],    # Row 1: Early to mid training
        [2, 6, 11, 16],    # Row 2: Early to mid training
        [3, 7, 12, 17],    # Row 3: Early to mid training
        [4, 8, 13, 18],    # Row 4: Early to mid training
        [5, 9, 14, 19],    # Row 5: Early to mid training
    ]
    
    # Alternative: Show key progression epochs
    key_epochs = [
        [1, 5, 10, 15],    # Row 1: Key progression points
        [2, 6, 11, 16],    # Row 2: Adjacent epochs
        [3, 7, 12, 17],    # Row 3: Adjacent epochs
        [4, 8, 13, 18],    # Row 4: Adjacent epochs
        [5, 9, 14, 19],    # Row 5: Adjacent epochs
    ]
    
    samples_dir = "runs/mnist_dcgan/samples"
    
    # Check if samples exist
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory {samples_dir} not found!")
        return
    
    # Load images
    images = []
    for row in key_epochs:
        row_images = []
        for epoch in row:
            img_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path)
                row_images.append(img)
            else:
                print(f"Warning: {img_path} not found!")
                # Create a placeholder image
                placeholder = Image.new('RGB', (280, 280), color='white')
                row_images.append(placeholder)
        images.append(row_images)
    
    # Calculate grid dimensions
    # Each sample image is 10x10 grid of 28x28 images = 280x280
    img_width, img_height = images[0][0].size
    grid_width = img_width * 4  # 4 columns
    grid_height = img_height * 5  # 5 rows
    
    # Create the final grid
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for row_idx, row_images in enumerate(images):
        for col_idx, img in enumerate(row_images):
            x = col_idx * img_width
            y = row_idx * img_height
            grid_image.paste(img, (x, y))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(grid_image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add epoch labels
    for row_idx, row_epochs in enumerate(key_epochs):
        for col_idx, epoch in enumerate(row_epochs):
            x = col_idx * img_width + 10
            y = row_idx * img_height + 10
            draw.text((x, y), f"Epoch {epoch}", fill='red', font=font)
    
    # Save the grid
    output_path = "training_progress_grid.png"
    grid_image.save(output_path)
    print(f"Training progress grid saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_training_progress_grid()
