from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm

from data.dataset import get_train_val_datasets, get_train_val_dataloaders
from model.vit import create_vit_model
from metrics import calculate_metrics
from utils import (
    load_config,
    set_seed,
    get_device,
    create_dir,
    save_history_csv,
    save_checkpoint,
    plot_training_curves,
    EarlyStopping,
)

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "configs" / "optuna_tuning.yaml")

def train_epoch(model, train_loader, criterion, optimizer, device, mixup_fn=None):
    model.train()

    running_loss = 0.0
    all_outputs = []
    all_targets = []

    progress_bar = tqdm(train_loader, desc="Treinando rede", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        labels_for_metrics = labels

        optimizer.zero_grad()

        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        all_outputs.append(outputs.detach().cpu())
        all_targets.append(labels_for_metrics.detach().cpu())

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader.dataset)
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    metrics = calculate_metrics(all_outputs, all_targets)

    return epoch_loss, metrics

def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validando rede", leave=False)

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            all_outputs.append(outputs.detach().cpu())
            all_targets.append(labels.detach().cpu())

    epoch_loss = running_loss / len(val_loader.dataset)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_outputs, all_targets)

    return epoch_loss, metrics

def main(config: dict):
    experiment_name = config["experiment"]["name"]
    seed = config["experiment"]["seed"]

    data_dir = config["paths"]["data_dir"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    results_dir = config["paths"]["results_dir"]

    image_size = config["dataset"]["image_size"]
    num_classes = config["dataset"]["num_classes"]
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    download = config["dataset"]["download"]
    use_augmentation = config["dataset"].get("augmentation", False)
    random_erasing_p = config["dataset"].get("random_erasing_p", 0.0)

    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]

    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]
    weight_decay = config["training"]["weight_decay"]
    label_smoothing = config["training"].get("label_smoothing", 0.0)
    mixup_alpha = config["training"].get("mixup_alpha", 0.0)
    early_stopping_config = config["training"].get("early_stopping", {})
    early_stopping_enabled = early_stopping_config.get("enabled", False)
    early_stopping_monitor = early_stopping_config.get("monitor", "val_accuracy")

    set_seed(seed)

    device = get_device()
    print(f"Experimento: {experiment_name}")
    print(f"Dispositivo usado: {device}")

    create_dir(checkpoint_dir)
    create_dir(results_dir)

    train_dataset, val_dataset = get_train_val_datasets(
        data_dir=data_dir,
        image_size=image_size,
        download=download,
        use_augmentation=use_augmentation,
        random_erasing_p=random_erasing_p,
    )

    train_loader, val_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Tamanho treino: {len(train_dataset)}")
    print(f"Tamanho validação: {len(val_dataset)}")

    model = create_vit_model(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
    )

    model = model.to(device)

    eval_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    train_criterion = eval_criterion
    mixup_fn = None

    if mixup_alpha > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=0.0,
            prob=1.0,
            switch_prob=0.0,
            mode="batch",
            label_smoothing=label_smoothing,
            num_classes=num_classes,
        )
        train_criterion = SoftTargetCrossEntropy()

        print(
            "MixUp habilitado | "
            f"alpha: {mixup_alpha} | "
            "prob: 1.0 | "
            "mode: batch"
        )

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    best_val_accuracy = 0.0

    checkpoint_path = Path(checkpoint_dir) / f"{experiment_name}_best.pth"
    early_stopping = None

    if early_stopping_enabled:
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get("patience", 7),
            min_delta=early_stopping_config.get("min_delta", 0.0),
            mode=early_stopping_config.get("mode", "max"),
        )

        print(
            "Early stopping habilitado | "
            f"monitor: {early_stopping_monitor} | "
            f"patience: {early_stopping.patience}"
        )

    for epoch in range(1, epochs + 1):
        print(f"\nÉpoca {epoch}/{epochs}")

        train_loss, train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            mixup_fn=mixup_fn,
        )

        val_loss, val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=eval_criterion,
            device=device,
        )

        scheduler.step()

        train_accuracy = train_metrics["accuracy"]
        val_accuracy = val_metrics["accuracy"]

        train_macro_f1 = train_metrics["macro_f1"]
        val_macro_f1 = val_metrics["macro_f1"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Train Macro-F1: {train_macro_f1:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_macro_f1"].append(val_macro_f1)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_val_accuracy,
                path=checkpoint_path,
            )

            print(f"Novo melhor modelo salvo em: {checkpoint_path}")

        monitored_values = {
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_loss": val_loss,
        }

        if early_stopping_monitor not in monitored_values:
            raise ValueError(
                "Monitor de early stopping invalido: "
                f"{early_stopping_monitor}. Use val_accuracy, val_macro_f1 ou val_loss."
            )

        if early_stopping is not None and early_stopping.step(monitored_values[early_stopping_monitor]):
            print(
                "Early stopping acionado: "
                f"{early_stopping.counter} epocas sem melhora em {early_stopping_monitor}."
            )
            break

    history_csv_path = Path(results_dir) / "training_history.csv"

    save_history_csv(history, history_csv_path)

    plot_training_curves(
        history=history,
        output_dir=results_dir,
    )

    print("\nTreinamento finalizado.")
    print(f"Melhor validation accuracy: {best_val_accuracy:.4f}")
    print(f"Histórico salvo em: {history_csv_path}")
    print(f"Gráficos salvos em: {results_dir}")
    
    return best_val_accuracy

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = DEFAULT_CONFIG_PATH
        
    config = load_config(config_path)

    main(config)

    

