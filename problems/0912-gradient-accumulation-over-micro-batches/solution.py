import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    # TODO: zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss
    number_of_microbatches = len(micro_batches)
    model.zero_grad()
    total_loss = 0
    for (x,y) in micro_batches:
        prediction = model(x)
        loss = criterion(prediction,y)
        (loss/number_of_microbatches).backward()
        total_loss+=loss.item()
    optimizer.step()

    return total_loss/number_of_microbatches
