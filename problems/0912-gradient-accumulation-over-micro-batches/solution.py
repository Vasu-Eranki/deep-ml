import torch

def accumulated_step(model, micro_batches, optimizer, criterion):
    # TODO: zero grads, accumulate over micro-batches with proper scaling, step once, return mean loss
    number_of_microbatches = len(micro_batches)
    model.zero_grad()
    loss_values =[]
    for (x,y) in micro_batches:
        prediction = model(x)
        loss = criterion(prediction,y)
        loss*=1/number_of_microbatches
        loss.backward()
        loss_values.append(number_of_microbatches*loss.detach().numpy())
    optimizer.step()

    return float(sum(loss_values)/number_of_microbatches)
