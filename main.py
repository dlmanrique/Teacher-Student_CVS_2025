import torch
import torch.nn as nn

from model import build_swinv2
from dataset import get_datasets, get_dataloader

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # logits sin sigmoid
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.float().to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = (all_preds == all_labels).float().mean().item()
    return avg_loss, acc



if __name__ == '__main__':

    fold=2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 3   # ajusta a tu caso
    model_name = "swinv2_tiny_window16_256"

    model = build_swinv2(model_name, num_labels=num_classes, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()  # multietiqueta binaria
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    ## Dataset and Dataloader
    train_dataset, val_dataset = get_datasets(fold)
    train_loader, val_loader = get_dataloader(train_dataset, val_dataset)

    # Loop de entrenamiento
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Época {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")