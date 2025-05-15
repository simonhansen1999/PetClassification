from torchvision import transforms
import torch
from sklearn.metrics import confusion_matrix, classification_report
from transformers import ViTForImageClassification, ViTImageProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import oxford_pet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

def collate_fn(batch):
    images, labels, targets = zip(*batch)
    return list(images), list(labels), list(targets)

from transformers import ViTForImageClassification, ViTImageProcessor

# Load pretrained ViT + update for binary classification
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    id2label={0: "cat", 1: "dog"},
    label2id={"cat": 0, "dog": 1},
    ignore_mismatched_sizes=True
)

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

from torch.utils.data import DataLoader

# Smaller subset for quick testing
train_dataset = oxford_pet.OxfordPetDetectionDataset("./", split="train", transforms=transform, max_size=100)
val_dataset   = oxford_pet.OxfordPetDetectionDataset("./", split="val", transforms=transform, max_size=100)
test_dataset  = oxford_pet.OxfordPetDetectionDataset("./", split="test", transforms=transform, max_size=100)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

img,label, target = train_dataset[0]
oxford_pet.plot_image_with_boxes(img, target)
img,label, target = train_dataset[10]
oxford_pet.plot_image_with_boxes(img, target)

import torch.optim as optim
import torch.nn as nn

for param in model.parameters():
    param.requires_grad = False

# Unfreeze only the classification head
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

for epoch in range(3):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels, targets in train_loader:  # Unpack batch directly
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images).logits
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total
print(f"\nTest Accuracy: {acc:.4f}")

from sklearn.metrics import confusion_matrix, classification_report

class_names=["Cat", "Dog"]
cm = confusion_matrix(all_labels,all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Optional detailed report
print(classification_report(all_labels,all_preds, target_names=["Cat", "Dog"]))

#clean up delete previous model
import gc
# Clean up model if it exists
if 'model' in locals():
    del model

# Clean up inputs if you're not sure they're defined
#if 'inputs' in locals():
#    del inputs

# Clear cache and run garbage collection
gc.collect()
torch.cuda.empty_cache()

import oxford_pet
import torch

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
#%matplotlib inline

def collate_fn(batch):
    images, labels, targets = zip(*batch)
    return list(images), list(labels), list(targets)

from transformers import DetrImageProcessor, DetrForObjectDetection
device = torch.device("cpu")
model_name = "facebook/detr-resnet-50"
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50",
    size={"shortest_edge": 800, "longest_edge": 1333},
    do_resize=True,
    do_rescale=True,
    do_normalize=True
)
detr_model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")

detr_model.eval()
detr_model.to(device)

detection_val = oxford_pet.OxfordPetDetectionDataset("./", split="test", transforms=None, max_size=10)

detection_loader = torch.utils.data.DataLoader(
    detection_val,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

oxford_pet.evaluate_detr(detr_model, processor, detection_loader, device, score_threshold=0.7)
oxford_pet.evaluate_detr(detr_model, processor, detection_loader,device, score_threshold=0.99)
oxford_pet.evaluate_detr(detr_model, processor, detection_loader, device, score_threshold=0.1)

img, _, target = detection_val[0]
oxford_pet.plot_detr_prediction(img, detr_model, processor, device, target=target, score_threshold=0.98)