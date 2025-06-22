import pandas as pd
import numpy as np
from PIL import Image
import os

def process_mnist_csv(csv_file_path, num_images=4):
    """
    process mnist csv data and convert to c arrays and images
    
    args:
        csv_file_path: path to the csv file containing mnist data
        num_images: number of images to process and save
    """
    
    # read csv file using pandas with explicit column handling
    # pandas automatically detects the comma delimiter and creates a dataframe
    df = pd.read_csv(csv_file_path)
    
    # extract labels from first column - these are the digit classifications (0-9)
    # .values converts pandas series to numpy array for faster array operations
    labels = df['label'].values
    
    # extract pixel data by dropping the label column
    # this creates a dataframe with only pixel columns (pixel0 through pixel783)
    pixel_data = df.drop('label', axis=1)
    
    # convert pixel dataframe to numpy array
    # shape will be (num_samples, 784) where 784 = 28 * 28 pixels
    pixel_array = pixel_data.values
    
    print(f"loaded {len(labels)} samples")
    print(f"pixel data shape: {pixel_array.shape}")
    print(f"labels: {labels}")
    
    # process the first num_images samples
    for i in range(min(num_images, len(labels))):
        
        # extract single image data from row i
        # pixel_row is 1d array of 784 values representing flattened 28x28 image
        pixel_row = pixel_array[i]
        current_label = labels[i]
        
        print(f"\nprocessing sample {i}: digit {current_label}")
        
        # reshape 1d array (784,) to 2d array (28, 28) for image representation
        # numpy reshapes by filling row by row: [0,1,2...27] becomes first row, [28,29,30...55] becomes second row, etc.
        image_2d = pixel_row.reshape(28, 28)
        
        # create pil image from numpy array
        # 'L' mode specifies 8-bit grayscale (luminance only)
        # pil expects values 0-255 which matches our pixel data range
        img = Image.fromarray(image_2d.astype(np.uint8), 'L')
        
        # save image with descriptive filename
        # format: digit_{label}_sample_{index}.png
        img_filename = f"digit_{current_label}_sample_{i}.png"
        img.save(img_filename)
        print(f"saved image: {img_filename}")
        
        # generate c array representation
        generate_c_array(pixel_row, current_label, i)

def generate_c_array(pixel_data, label, sample_index):
    """
    convert single mnist sample to c array format
    
    args:
        pixel_data: 1d numpy array of 784 pixel values
        label: digit label (0-9)
        sample_index: index of current sample for naming
    """
    
    # create c array header comment
    c_code = f"// mnist digit {label} - sample {sample_index}\n"
    c_code += f"// 28x28 grayscale image flattened to 784 elements\n"
    c_code += f"// pixel values range from 0 (black) to 255 (white)\n\n"
    
    # declare c array with explicit size
    # unsigned char is 8-bit type perfect for 0-255 pixel values
    array_name = f"mnist_digit_{label}_sample_{sample_index}"
    c_code += f"unsigned char {array_name}[784] = {{\n"
    
    # format pixel values in rows of 28 to match original image structure
    # this makes the c array more readable and debuggable
    for row in range(28):
        c_code += "    "  # indentation for readability
        
        # extract 28 pixels for current row
        # row * 28 calculates starting index for each row
        row_start = row * 28
        row_end = row_start + 28
        row_pixels = pixel_data[row_start:row_end]
        
        # format each pixel value with consistent width
        # {:3d} ensures each number takes exactly 3 characters (right-aligned)
        pixel_strings = [f"{pixel:3d}" for pixel in row_pixels]
        
        # join pixels with commas, add trailing comma except for last row
        c_code += ", ".join(pixel_strings)
        
        if row < 27:  # not the last row
            c_code += ","
        
        c_code += "\n"
    
    c_code += "};\n\n"
    
    # add helper information
    c_code += f"// label for this data: {label}\n"
    c_code += f"// image dimensions: 28 x 28 pixels\n"
    c_code += f"// total elements: {len(pixel_data)}\n"
    
    # save c array to file
    c_filename = f"mnist_digit_{label}_sample_{sample_index}.c"
    with open(c_filename, 'w') as f:
        f.write(c_code)
    
    print(f"saved c array: {c_filename}")
    
    # print first few elements for verification
    print(f"first 10 pixel values: {pixel_data[:10]}")
    print(f"array name: {array_name}")

def analyze_pixel_distribution(csv_file_path):
    """
    analyze the distribution of pixel values across all samples
    """
    
    df = pd.read_csv(csv_file_path)
    pixel_data = df.drop('label', axis=1).values
    
    # flatten all pixel data into single array for analysis
    all_pixels = pixel_data.flatten()
    
    print(f"\npixel value analysis:")
    print(f"total pixels: {len(all_pixels)}")
    print(f"min value: {np.min(all_pixels)}")
    print(f"max value: {np.max(all_pixels)}")
    print(f"mean value: {np.mean(all_pixels):.2f}")
    print(f"std deviation: {np.std(all_pixels):.2f}")
    
    # count non-zero pixels (helps understand sparsity)
    non_zero_count = np.count_nonzero(all_pixels)
    zero_count = len(all_pixels) - non_zero_count
    print(f"non-zero pixels: {non_zero_count} ({non_zero_count/len(all_pixels)*100:.1f}%)")
    print(f"zero pixels: {zero_count} ({zero_count/len(all_pixels)*100:.1f}%)")

# usage example
if __name__ == "__main__":
    # process the mnist csv file
    # change this path to match your actual file location
    csv_path = "train.csv"  # or "paste.txt" depending on your file
    
    try:
        # main processing function
        process_mnist_csv(csv_path, num_images=20)
        
        # additional analysis
        analyze_pixel_distribution(csv_path)
        
        print(f"\nprocessing complete!")
        print(f"check current directory for generated files:")
        print(f"- *.png files: grayscale images of digits")
        print(f"- *.c files: c array representations")
        
    except FileNotFoundError:
        print(f"error: could not find file {csv_path}")
        print(f"make sure the csv file is in the current directory")
    except Exception as e:
        print(f"error during processing: {e}")