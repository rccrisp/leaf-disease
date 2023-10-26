import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def create_image_grid(image_data, n_images_per_dir=3, scale=3):

    fig, axs = plt.subplots(len(image_data), n_images_per_dir+1, figsize=(n_images_per_dir*scale, len(image_data)*scale))

    plt.rcParams["font.family"] = "serif"
    for i, (label, directory) in enumerate(image_data.items()):
        # Display the label on the far left
        axs[i, 0].text(0.5, 0.5, label, ha='left', va='center', fontsize=12)
        axs[i, 0].axis('off')

        image_files = [os.path.join(directory, filename) for filename in os.listdir(directory)[:n_images_per_dir]]
        
        for j, image_file in enumerate(image_files):
            image = Image.open(image_file)
            axs[i, j + 1].imshow(image)
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()

def plot_class_distribution(class_counts, size=20):

    # Extract class names and counts from the dictionary
    class_names = list(class_counts.keys())
    class_values = list(class_counts.values())

    def my_fmt(x):
        print(x)
        return '{:.2f}%\n({:.0f})'.format(x, total*x/100)
   
    total = sum(class_values)

    plt.figure(figsize=(8, 8))
    plt.rcParams["font.family"] = "serif"
    plt.pie(class_values, labels=class_names, autopct=my_fmt, startangle=140, textprops={'fontsize': size})
    plt.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.

    plt.title('')
    plt.show()



def count_image_files_in_subdirectories(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.JPG')

    result = {}

    for root, dirs, files in os.walk(directory):
        image_count = 0

        for file in files:
            if file.lower().endswith(image_extensions):
                image_count += 1
        if image_count > 0:
            result[os.path.basename(root)] = image_count

    return result

def create_bar_plot(data, title="", x_label="", y_label=""):
    # Extract keys (categories) and values (quantities) from the dictionary
    categories = list(data.keys())
    quantities = list(data.values())

    # Create a bar plot
    plt.figure(figsize=(12, 8))
    plt.rcParams["font.family"] = "serif"
    plt.bar(categories, quantities)
    
    # Rotate x-axis labels vertically
    plt.xticks(categories, rotation=90)

    # Add title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Display the plot
    plt.show()

def calculate_vegetation_index(image):
    # Extract the red, green, and blue channels
    r, g, b = image[0, :, :], image[1, :, :,], image[2, :, :]

    # Calculate the vegetation index for each pixel
    vegetation_index = 0.441 * r - 0.881 * g + 0.385 * b + 18.787

    return vegetation_index



