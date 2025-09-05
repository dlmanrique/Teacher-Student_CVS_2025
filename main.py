import torch
import torch.nn as nn
import wandb
import argparse
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import os, random, numpy as np


from model import build_swinv2, build_teacher
from dataset import SwinDataset, get_dataloader
from loss import DistillationLoss
from evaluation import get_map

torch.set_num_threads(1)

seed = 5
# Environment Standardisation
random.seed(seed)                      # Set random seed
np.random.seed(seed)                   # Set NumPy seed
torch.manual_seed(seed)                # Set PyTorch seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seed)           # Set CUDA seed
torch.use_deterministic_algorithms(True) # Force deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # CUDA workspace config



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    print('Training ....')
    for imgs, labels, _, _ in tqdm(dataloader):
        imgs, labels = imgs.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(imgs)  # logits sin sigmoid
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        wandb.log({'Train loss': loss.item()})
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def train_teacher_student(model, teacher, dataloader, device, args):
    model.train()
    teacher.eval()

    # class weights y loss function
    class_weights = torch.tensor([[3.19852941, 4.46153846, 2.79518072]]).to(device)
    criterion = DistillationLoss(class_weights, T=args.T, alpha=args.alpha, device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Training teacher-student ....')
    total_loss = 0.0

    for imgs_student, labels, imgs_teacher, _ in tqdm(dataloader):
        imgs_student = imgs_student.to(device)
        imgs_teacher = imgs_teacher.to(device)
        labels = labels.float().to(device)

        # forward teacher (sin gradiente)
        with torch.no_grad():
            teacher_logits = teacher(imgs_teacher)

        # forward student
        student_logits = model(imgs_student)

        # loss
        loss = criterion(student_logits, teacher_logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Train Loss: {avg_loss:.4f}")

    return avg_loss






def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_running_loss = 0.0
    val_probabilities = []
    val_predictions = []
    val_targets = []

    print('Eval ....')
    with torch.no_grad():
        for imgs, labels, _, _ in tqdm(dataloader):
            imgs, labels = imgs.to(device), labels.float().to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            # Get outputs
            val_probability = torch.sigmoid(outputs)
            val_prediction = torch.round(val_probability)

            val_probabilities.append(val_probability.to('cpu'))
            val_predictions.append(val_prediction.to('cpu'))
            val_targets.append(labels.to('cpu'))

            val_running_loss += loss.item() * imgs.size(0)
            wandb.log({'Val loss': loss.item()})

    val_loss = val_running_loss / len(dataloader.dataset)
    C1_ap, C2_ap, C3_ap, mAP = get_map(val_targets, val_probabilities)
    print('mAP', round(mAP, 4))
    print('C1 ap', round(C1_ap, 4))
    print('C2 ap', round(C2_ap, 4))
    print('C3 ap', round(C3_ap, 4))
    wandb.log({'mAP': mAP, 'C1': C1_ap, 'C2': C2_ap, 'C3': C3_ap})

    return val_loss, C1_ap, C2_ap, C3_ap, mAP




if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Run SwinCVS with specified config")
    parser.add_argument('--fold', type=int, default=2, required=False)
    parser.add_argument('--lr', type=float, default=0.00000464158883) # IdeaL: 0.00000464158883
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64) # Ideal: 16
    parser.add_argument('--training_type', type=str, default='Base', choices=['Base', 'Teacher-Student'])
    parser.add_argument('--model_name', type=str, default='swinv2_base_window8_256', choices=['swinv2_base_window8_256', 'swinv2_small_window16_256'])
    parser.add_argument('--teacher_path', type=str, default='weights/Sages_Fold2_bestMAP.pt')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--T', type=float, default=2)
    args = parser.parse_args()
    
    # Wandb log
    exp_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    wandb.init(
        project='Swin_teacher_student', 
        entity='endovis_bcv',
        config=vars(args),
        name=exp_name 
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 3   # ajusta a tu caso
    model_name = args.model_name # or "swinv2_small_window16_256" 

    ## Dataset and Dataloader
    train_dataset = SwinDataset(args.fold, 'train')
    val_dataset = SwinDataset(args.fold, 'test')

    train_loader, val_loader = get_dataloader(train_dataset, val_dataset, args.batch_size)

    model = build_swinv2(model_name, num_labels=num_classes, pretrained=True).to(device)

    class_weights = torch.tensor([[3.19852941, 4.46153846, 2.79518072]]).to('cuda')
    criterion = nn.BCEWithLogitsLoss(weight=class_weights).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_mAP = 0
    # Loop de entrenamiento
    for epoch in range(args.epochs):
        print(f'Epoch [{epoch}/{args.epochs}]')
        if args.training_type == 'Base':
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        elif args.training_type == 'Teacher-Student':
            # Load teacher model
            teacher = build_teacher(args.teacher_path)
            teacher.to('cuda')
            train_loss = train_teacher_student(model, teacher, train_loader, device, args)
        val_loss, C1_ap, C2_ap, C3_ap, mAP = evaluate(model, val_loader, criterion, device)

        print(f"Época {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val mAP={mAP:.4f}")

        if mAP > best_mAP:
            best_mAP = mAP
            best_C1 = C1_ap
            best_C2 = C2_ap
            best_C3 = C3_ap
            save_folder = '/media/lambda001/SSD3/dlmanrique/Endovis/CVS_Challenge/Teacher_student_results'
            # Save model weights
            os.makedirs(f'{save_folder}/Fold{args.fold}/{exp_name}', exist_ok=True)
            torch.save(model.state_dict(), f'{save_folder}/Fold{args.fold}/{exp_name}/best.pt')
            wandb.log({'Best mAP': best_mAP, 'Best C1':best_C1, 'Best C2':best_C2, 'Best C3': best_C3})

print('--------Best results-----------------')
print('Best mAP', round(best_mAP, 4))
print('Best C1 ap', round(best_C1, 4))
print('Best C2 ap', round(best_C2, 4))
print('Best C3 ap', round(best_C3, 4))