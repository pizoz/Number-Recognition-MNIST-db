import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor, Resize, Grayscale

# -------------------
# Load the trained model
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_state_dict.pth", map_location=device))
model.eval()

# Labels mapping
labels_map = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
    5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
}

# -------------------
# Tkinter Drawing UI
# -------------------

img_size = 280
canvas = Image.new("L", (img_size, img_size), 0)  # Black canvas
draw = ImageDraw.Draw(canvas)
last_x, last_y = None, None  # Track previous mouse position
line_width = 20

def start_draw(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y
    # Draw initial point
    param = 10  # Matches line width for consistency
    draw.ellipse([event.x-param, event.y-param, event.x+param, event.y+param], fill=255)
    canvas_tk.create_oval(event.x-param, event.y-param, event.x+param, event.y+param, fill="white")

def draw_digit(event):
    global last_x, last_y
    if last_x is None or last_y is None:
        start_draw(event)
        return
    
    # Calculate intermediate points for smooth drawing
    distance = max(abs(event.x - last_x), abs(event.y - last_y))
    if distance == 0:
        return
    
    param = line_width // 2
    
    # Draw along the path between points
    for i in range(distance + 1):
        x = last_x + (event.x - last_x) * i / distance
        y = last_y + (event.y - last_y) * i / distance
        draw.ellipse([x-param, y-param, x+param, y+param], fill=255)
        canvas_tk.create_oval(x-param, y-param, x+param, y+param, fill="white", outline="white")
    
    last_x, last_y = event.x, event.y


def preprocess_image(img):
    """Convert the drawn image to a 28x28 tensor suitable for the model"""
    img = img.resize((28, 28))  # Resize to MNIST size
    img.show()      # Convert to grayscale
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    #img_tensor = 1.0 - img_tensor  # Invert colors (black background becomes white)
    print(img_tensor)
    return img_tensor

def predict():
    """Process the image and make a prediction"""
    img_tensor = preprocess_image(canvas)

    # Make prediction
    with torch.no_grad():
        pred_logits = model(img_tensor)
        predicted_label = pred_logits.argmax(1).item()

    # Show the result
    result_label.config(text=f"Predicted: {labels_map[predicted_label]}")

def clear_canvas():
    """Clear the drawing canvas"""
    global canvas, draw, last_x, last_y
    canvas = Image.new("L", (img_size, img_size), 0)  # Reset to black canvas
    draw = ImageDraw.Draw(canvas)
    canvas_tk.delete("all")
    result_label.config(text="Draw a number")
    last_x, last_y = None, None  # Reset position tracker

# -------------------
# Tkinter UI Setup
# -------------------

root = tk.Tk()
root.title("Draw a Number")

canvas_tk = tk.Canvas(root, width=img_size, height=img_size, bg="black")
canvas_tk.pack()

# Bind mouse events
canvas_tk.bind("<Button-1>", start_draw)
canvas_tk.bind("<B1-Motion>", draw_digit)

btn_frame = tk.Frame(root)
btn_frame.pack()

predict_btn = tk.Button(btn_frame, text="Predict", command=predict)
predict_btn.pack(side=tk.LEFT)

clear_btn = tk.Button(btn_frame, text="Clear", command=clear_canvas)
clear_btn.pack(side=tk.RIGHT)

result_label = tk.Label(root, text="Draw a number", font=("Arial", 16))
result_label.pack()

root.mainloop()