import matplotlib.pyplot  as plt
import os
import torch
from torch import nn
import torchvision
from torchvision.transforms import ToTensor
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),  
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_state_dict.pth"))
model.eval()
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
figure = plt.figure(figsize=(8, 8))
cols, rows = 4, 4
labels_map = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}

# Disattiva il calcolo dei gradienti durante l'inferenza
with torch.no_grad():
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(test_data), size=(1,)).item()
        img, label = test_data[sample_idx]
        
        # Aggiungi una dimensione batch e sposta l'immagine sul device corretto
        img_input = img.unsqueeze(0).to(device)
        
        # Ottieni la predizione dal modello
        pred_logits = model(img_input)
        predicted_label = pred_logits.argmax(1).item()
        
        # Aggiungi il subplot con titolo che mostra etichetta reale e predetta
        ax = figure.add_subplot(rows, cols, i)
        ax.set_title(f"Real value: {labels_map[label]}\nPrediction: {labels_map[predicted_label]}")
        ax.axis("off")
        ax.imshow(img.squeeze(), cmap="gray")
        
plt.show(block=True)