import cv2
import numpy as np
import torch
import torch.nn as nn
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.png")])

if not file_path:
    exit()

img = cv2.imread(file_path)

grid_size = 50
color = (0, 255, 0)  # Green
# Create an array with zeros (black image) of the  shape as the loaded image using PyTorch
grid_img = torch.zeros_like(torch.from_numpy(img)).numpy()


grid_img[:, ::grid_size] = color

grid_img[::grid_size, :] = color

result = cv2.addWeighted(img, 1, grid_img, 1, 0)

# Display the image with the grid using Seaborn
sns.set()
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
ax.set_title('Image with Grid')
ax.axis('off')
plt.show()

root = tk.Tk()
root.title("Grid Drawer")

def draw_grid():
    img_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.jpg;*.png")])
    if not img_path:
        return

    img = cv2.imread(img_path)

    grid_img = torch.zeros_like(torch.from_numpy(img)).numpy()

   
    grid_img[:, ::grid_size] = color

    grid_img[::grid_size, :] = color

   
    result = cv2.addWeighted(img, 1, grid_img, 1, 0)

  
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax.set_title('Image with Grid')
    ax.axis('off')
    plt.show()

button = tk.Button(root, text="Draw Grid on Image", command=draw_grid)
button.pack(pady=20)

root.mainloop()
