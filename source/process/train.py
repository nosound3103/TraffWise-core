import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics
from torchmetrics import MeanAveragePrecision
from tqdm import tqdm
from typing import Dict, List


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               epoch: int,
               num_epochs: int = 10):
    model.train()
    rpn_classification_losses = []
    rpn_localization_losses = []
    frcnn_classification_losses = []
    frcnn_localization_losses = []

    with tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}: Training") as pbar:
        for images, targets in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]
            loss_dict = model(images, targets)

            loss = loss_dict['loss_classifier']
            loss += loss_dict['loss_box_reg']
            loss += loss_dict['loss_rpn_box_reg']
            loss += loss_dict['loss_objectness']

            rpn_classification_losses.append(
                loss_dict['loss_objectness'].item())
            rpn_localization_losses.append(
                loss_dict['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(
                loss_dict['loss_classifier'].item())
            frcnn_localization_losses.append(
                loss_dict['loss_box_reg'].item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)

    rpn_classification_loss = np.mean(rpn_classification_losses)
    rpn_localization_loss = np.mean(rpn_localization_losses)
    frcnn_classification_loss = np.mean(frcnn_classification_losses)
    frcnn_localization_loss = np.mean(frcnn_localization_losses)

    return rpn_classification_loss, \
        rpn_localization_loss, \
        frcnn_classification_loss, \
        frcnn_localization_loss


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             metric: torchmetrics.Metric,
             device: torch.device,
             epoch: int,
             num_epochs: int = 10):

    model.train()

    metric.reset()

    rpn_classification_losses = []
    rpn_localization_losses = []
    frcnn_classification_losses = []
    frcnn_localization_losses = []
    with torch.no_grad():
        with tqdm(
                total=len(dataloader),
                desc=f"Epoch {epoch+1}/{num_epochs}: Validation") as pbar:

            for images, targets in dataloader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in targets]
                loss_dict = model(images, targets)

                loss = loss_dict['loss_classifier']
                loss += loss_dict['loss_box_reg']
                loss += loss_dict['loss_rpn_box_reg']
                loss += loss_dict['loss_objectness']

                rpn_classification_losses.append(
                    loss_dict['loss_objectness'].item())
                rpn_localization_losses.append(
                    loss_dict['loss_rpn_box_reg'].item())
                frcnn_classification_losses.append(
                    loss_dict['loss_classifier'].item())
                frcnn_localization_losses.append(
                    loss_dict['loss_box_reg'].item())

                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

    model.eval()
    for images, targets in dataloader:
        images = list(image.to(device) for image in images)
        preds = model(images)
        preds_formatted = [
            {"boxes": p["boxes"].cpu(),
             "labels": p["labels"].cpu(),
             "scores": p["scores"].cpu()}
            for p in preds]

        metric.update(preds=preds_formatted, target=targets)

    results = metric.compute()
    map50 = results["map_50"].item()
    map95 = results["map"].item()

    rpn_classification_loss = np.mean(rpn_classification_losses)
    rpn_localization_loss = np.mean(rpn_localization_losses)
    frcnn_classification_loss = np.mean(frcnn_classification_losses)
    frcnn_localization_loss = np.mean(frcnn_localization_losses)

    return rpn_classification_loss, \
        rpn_localization_loss, \
        frcnn_classification_loss, \
        frcnn_localization_loss, \
        map50, map95


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          epochs: int) -> Dict[str, List]:

    results = {
        "train_rpn_cls_loss": [],
        "train_rpn_local_loss": [],
        "train_frcnn_cls_loss": [],
        "train_frcnn_local_loss": [],
        "val_rpn_cls_loss": [],
        "val_rpn_local_loss": [],
        "val_frcnn_cls_loss": [],
        "val_frcnn_local_loss": [],
        "val_map": [],
        "val_map95": []
    }

    map_metric = MeanAveragePrecision()

    for epoch in range(epochs):
        t_rpn_c_l, t_rpn_l_l, \
            t_frcnn_c_l, t_frcnn_l_l = train_step(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=epochs)

        v_rpn_c_l, v_rpn_l_l, v_frcnn_c_l, \
            v_frcnn_l_l, v_map, v_map95 = val_step(
                model=model,
                dataloader=val_dataloader,
                device=device,
                metric=map_metric,
                epoch=epoch,
                num_epochs=epochs)

        loss_output = ''
        loss_output += f'Epoch: {epoch+1}\n'
        loss_output += f'Train RPN Classification Loss : {t_rpn_c_l:.4f}\n'
        loss_output += f'Train RPN Localization Loss : {t_rpn_l_l:.4f}\n'
        loss_output += f'Train FRCNN Classification Loss : {t_frcnn_c_l:.4f}\n'
        loss_output += f'Train FRCNN Localization Loss : {t_frcnn_l_l:.4f}\n'
        loss_output += f'Val RPN Classification Loss : {v_rpn_c_l:.4f}\n'
        loss_output += f'Val RPN Localization Loss : {v_rpn_l_l:.4f}\n'
        loss_output += f'Val FRCNN Classification Loss : {v_frcnn_c_l:.4f}\n'
        loss_output += f'Val FRCNN Localization Loss : {v_frcnn_l_l:.4f}\n'
        loss_output += f'Val mAP50 : {v_map:.4f}\n'
        loss_output += f'Val mAP95 : {v_map95:.4f}\n'

        print(loss_output)

        results["train_rpn_cls_loss"].append(t_rpn_c_l)
        results["train_rpn_local_loss"].append(t_rpn_l_l)
        results["train_frcnn_cls_loss"].append(t_frcnn_c_l)
        results["train_frcnn_local_loss"].append(t_frcnn_l_l)
        results["val_rpn_cls_loss"].append(v_rpn_c_l)
        results["val_rpn_local_loss"].append(v_rpn_l_l)
        results["val_frcnn_cls_loss"].append(v_frcnn_c_l)
        results["val_frcnn_local_loss"].append(v_frcnn_l_l)
        results["val_map"].append(v_map)
        results["val_map95"].append(v_map95)

    return results
