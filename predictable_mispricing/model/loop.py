# %%
# Packages
import pandas as pd
import torch


# %%
# Functions
def train_epoch(dataloader, model, loss_fn, optimizer, scheduler):
    model.train()  # variable dropout + normalization layers
    running_loss = 0.0
    running_target_sq = 0.0
    running_N = 0

    for char, target, weight, _, _ in dataloader:
        optimizer.zero_grad(set_to_none=True)  # speeds up comp.

        exp = model.forward(char=char)
        loss = loss_fn(input=exp, target=target, weight=weight)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()  # for OneCycle

        # loss stuff
        running_loss += loss.item() * len(target)
        running_N += len(target)
        running_target_sq += (target.pow(2) * weight).sum().item()

    return running_loss / running_N, 1 - running_loss / running_target_sq


def validate_epoch(dataloader, model, loss_fn):
    model.eval()  # fixed dropout + normalization layers
    running_loss = 0.0
    running_target_sq = 0.0
    running_N = 0

    with torch.no_grad():
        for char, target, weight, _, _ in dataloader:
            exp = model.forward(char=char)
            loss = loss_fn(input=exp, target=target, weight=weight)

            # loss stuff
            running_loss += loss.item() * len(target)
            running_N += len(target)
            running_target_sq += (target.pow(2) * weight).sum().item()

    return running_loss / running_N, 1 - running_loss / running_target_sq


def testing_epoch(dataloader, model):
    model.eval()  # fixed dropout + normalization layers

    info_dict = dataloader._dataloader.dataset.retrieve_info()

    outputs = []
    with torch.no_grad():
        for char, target, _, dates, ids in dataloader:
            exp = model.forward(char=char)

            out = pd.DataFrame(info_dict["dates"][dates.cpu().long().numpy()], columns=["date"])
            out["id"] = info_dict["ids"][ids.cpu().long().numpy()]
            out["target"] = target.detach().cpu().numpy()
            out["pred"] = exp.detach().cpu().numpy()

            out = out.set_index(["date", "id"])
            outputs.append(out)

    outputs = pd.concat(outputs).sort_index()

    return outputs


# %%
