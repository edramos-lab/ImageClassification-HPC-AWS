import os
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset, DistributedSampler, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
import mlflow
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchmetrics.classification import Accuracy, Precision, F1Score, MatthewsCorrCoef
import cv2

# --- ALBUMENTATIONS IMPORTS ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURACIÓN DDP ---
def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
    else:
        print("ADVERTENCIA: Ejecutando en modo local (sin DDP).")
        rank = 0
        local_rank = 0
        world_size = 1
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

# --- CUSTOM DATASET PARA ALBUMENTATIONS ---
class AlbumentationsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(AlbumentationsImageFolder, self).__init__(root, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations usa RGB

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented["image"]
        
        return image, target

# --- DEFINICIÓN DE TRANSFORMACIONES ---
def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(224, 224),
            
            # 1. Flip (Invarianza rotacional básica)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # 2. ShiftScaleRotate (Simula posición y zoom)
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5),
            
            # 3. Color/Tinción (Simula variaciones de tinción H&E/Giemsa)
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            
            # 4. Brillo/Contraste (Simula iluminación del microscopio)
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # 5. Blur (Simula desenfoque de lente)
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            
            # 6. CoarseDropout (Simula artefactos/polvo y regulariza)
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.3),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        # Validación/Test: Solo Resize y Normalize
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# --- MODELOS CUSTOM (Igual que antes) ---
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes):
        super(HybridCNNViT, self).__init__()
        backbone = timm.create_model('resnet18', pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Conv2d(512, 512, kernel_size=1)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :])

def get_model(model_name, num_classes):
    if model_name == 'convnextv1':
        model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)
        target_layer = model.stages[-1].blocks[-1]
    elif model_name == 'custom_cnn':
        model = CustomCNN(num_classes)
        target_layer = model.features[-1]
    elif model_name == 'hybrid_vit':
        model = HybridCNNViT(num_classes)
        target_layer = model.features[-1]
    else:
        raise ValueError("Modelo desconocido")
    return model, target_layer

# --- LOOPS DE ENTRENAMIENTO ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    metrics = torch.tensor([running_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0] / metrics[2], (metrics[1] / metrics[2]).item()

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss, count = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            count += labels.size(0)
            all_preds.append(torch.softmax(outputs, dim=1).cpu())
            all_labels.append(labels.cpu())
    
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
    else:
        all_preds, all_labels = torch.tensor([]), torch.tensor([])
    return (running_loss / count if count > 0 else 0.0), all_preds, all_labels

# --- MAIN ---
def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    data_dir = "/home/ubuntu/data"
    
    # IMPORTANTE: Cargamos el dataset base sin transformaciones primero para obtener targets y clases
    # Las transformaciones se aplicarán dinámicamente al crear los Subsets
    try:
        # Usamos ImageFolder estándar solo para leer estructura
        base_dataset = datasets.ImageFolder(root=data_dir)
        targets = np.array(base_dataset.targets)
        classes = base_dataset.classes
    except Exception as e:
        if rank == 0: print(f"Error cargando dataset: {e}")
        return

    # Split Train/Test
    train_idx, test_idx = train_test_split(
        np.arange(len(targets)), test_size=0.1, stratify=targets, random_state=42
    )
    
    # Grid Search
    params_grid = {
        'batch_size': [16, 32],
        'dataset_ratio': [0.1, 0.25, 0.75, 1.0],
        'k_folds': [2, 5],
        'lr': [1e-3, 1e-5],
        'epochs': [10, 20],
        'optimizer': ['adam', 'adamw', 'adagrad'],
        'model_name': ['convnextv1', 'custom_cnn', 'hybrid_vit']
    }
    
    keys, values = zip(*params_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if rank == 0:
        mlflow.set_tracking_uri("file:///home/ubuntu/mlruns")
        mlflow.set_experiment("Leukemia_Albumentations_8GPU")

    for exp_idx, exp in enumerate(experiments):
        if rank == 0: print(f"--- Exp {exp_idx+1}/{len(experiments)} ---")
        
        # Subsampling
        if exp['dataset_ratio'] < 1.0:
            subset_idx, _ = train_test_split(
                train_idx, train_size=exp['dataset_ratio'], 
                stratify=targets[train_idx], random_state=42
            )
        else:
            subset_idx = train_idx
            
        skf = StratifiedKFold(n_splits=exp['k_folds'], shuffle=True, random_state=42)
        current_train_targets = targets[subset_idx]
        
        for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(subset_idx, current_train_targets)):
            
            # Índices
            abs_train_idx = subset_idx[train_fold_idx]
            abs_val_idx = subset_idx[val_fold_idx]
            
            # --- CREACIÓN DE DATASETS CON ALBUMENTATIONS ---
            # Aquí instanciamos AlbumentationsImageFolder y le pasamos los índices correctos
            # Nota: Subset de PyTorch no propaga transformaciones si el padre no las tiene.
            # Truco: Creamos dos instancias "físicas" del dataset completo con diferentes transforms
            
            train_full_ds = AlbumentationsImageFolder(root=data_dir, transform=get_transforms('train'))
            val_full_ds = AlbumentationsImageFolder(root=data_dir, transform=get_transforms('val'))
            
            train_set = Subset(train_full_ds, abs_train_idx)
            val_set = Subset(val_full_ds, abs_val_idx)
            test_set = Subset(val_full_ds, test_idx) # Test usa transformaciones de val (sin augment)

            # Samplers
            train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)
            
            train_loader = DataLoader(train_set, batch_size=exp['batch_size'], sampler=train_sampler, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=exp['batch_size'], sampler=val_sampler, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=exp['batch_size'], shuffle=False, num_workers=4)

            # Modelo
            model, target_layer = get_model(exp['model_name'], len(classes))
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=(exp['model_name']=='hybrid_vit'))
            
            # Optimizador
            if exp['optimizer'] == 'adam': opt = optim.Adam(model.parameters(), lr=exp['lr'])
            elif exp['optimizer'] == 'adamw': opt = optim.AdamW(model.parameters(), lr=exp['lr'])
            elif exp['optimizer'] == 'adagrad': opt = optim.Adagrad(model.parameters(), lr=exp['lr'])
            criterion = nn.CrossEntropyLoss()

            if rank == 0:
                run_name = f"{exp['model_name']}_b{exp['batch_size']}_fold{fold}_albumentations"
                mlflow.start_run(run_name=run_name)
                mlflow.log_params(exp)

            # Training loop
            for epoch in range(exp['epochs']):
                train_sampler.set_epoch(epoch)
                t_loss, t_acc = train_one_epoch(model, train_loader, criterion, opt, device)
                v_loss, v_preds, v_labels = evaluate(model, val_loader, criterion, device)
                
                v_acc = (v_preds.argmax(1) == v_labels).float().mean().item() if len(v_labels) > 0 else 0
                
                if rank == 0:
                    print(f"Epoch {epoch} | T_Loss: {t_loss:.3f} | T_Acc: {t_acc:.3f} | V_Loss: {v_loss:.3f} | V_Acc: {v_acc:.3f}")
                    mlflow.log_metrics({
                        "train_loss": t_loss, "train_acc": t_acc,
                        "val_loss": v_loss, "val_acc": v_acc
                    }, step=epoch)

            # Evaluación Final + Guardado de Modelo (Solo Rank 0)
            if rank == 0:
                model_eval = model.module
                model_eval.eval()

                # --- GUARDAR MODELO (.pth) ---
                models_dir = "/home/ubuntu/models"
                os.makedirs(models_dir, exist_ok=True)
                model_filename = f"{exp['model_name']}_exp{exp_idx}_fold{fold}.pth"
                model_path = os.path.join(models_dir, model_filename)
                torch.save({
                    'model_state_dict': model_eval.state_dict(),
                    'experiment': exp,
                    'fold': fold,
                    'classes': classes,
                }, model_path)
                print(f">>> Model saved: {model_path}")
                mlflow.log_artifact(model_path)

                # --- EVALUACIÓN EN TEST SET ---
                _, test_probs, test_labels = evaluate(model_eval, test_loader, criterion, device)
                
                if len(test_labels) > 0:
                    preds = test_probs.argmax(1).cpu()
                    lbls = test_labels.cpu()
                    num_cls = len(classes)
                    
                    acc = Accuracy(task="multiclass", num_classes=num_cls)
                    prec = Precision(task="multiclass", num_classes=num_cls, average='macro')
                    f1 = F1Score(task="multiclass", num_classes=num_cls, average='macro')
                    mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_cls)
                    
                    test_metrics = {
                        "test_acc": acc(preds, lbls).item(),
                        "test_precision": prec(preds, lbls).item(),
                        "test_f1": f1(preds, lbls).item(),
                        "test_mcc": mcc(preds, lbls).item(),
                    }
                    mlflow.log_metrics(test_metrics)
                    print(f">>> Test metrics: {test_metrics}")

                    # --- CONFUSION MATRIX ---
                    try:
                        cm = confusion_matrix(lbls.numpy(), preds.numpy())
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=classes, yticklabels=classes, ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        ax.set_title(f'Confusion Matrix - {run_name}')
                        cm_path = f"confusion_matrix_exp{exp_idx}_fold{fold}.png"
                        fig.savefig(cm_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        mlflow.log_artifact(cm_path)
                    except Exception as e:
                        print(f"Confusion matrix error: {e}")

                    # --- GRADCAM ---
                    try:
                        test_iter = iter(test_loader)
                        imgs, _ = next(test_iter)
                        imgs = imgs[:5].to(device)
                        cam = GradCAM(model=model_eval, target_layers=[target_layer])
                        grayscale_cam = cam(input_tensor=imgs)
                        
                        for i in range(len(imgs)):
                            img = imgs[i].cpu().permute(1, 2, 0).numpy()
                            img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                            img = np.clip(img, 0, 1)
                            viz = show_cam_on_image(img, grayscale_cam[i, :], use_rgb=True)
                            gradcam_path = f"gradcam_exp{exp_idx}_fold{fold}_img{i}.jpg"
                            plt.imsave(gradcam_path, viz)
                            mlflow.log_artifact(gradcam_path)
                    except Exception as e:
                        print(f"GradCAM error: {e}")

                mlflow.end_run()
            dist.barrier()

    cleanup_ddp()

if __name__ == '__main__':
    main()