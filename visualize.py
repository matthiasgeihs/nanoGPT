import torch, math
import matplotlib.pyplot as plt

def plot_tensor(tensor: torch.Tensor):
    # make tensor a single vector
    tensor = tensor.view(-1)
    
    # make square view
    n = int(math.ceil(math.sqrt(tensor.numel())))
    tensor = torch.nn.functional.pad(tensor, (0, n**2-tensor.numel()))
    tensor = tensor.view(n, n)

    # plot
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor.cpu())
    # colorbar
    plt.colorbar()
    # no grid
    plt.xticks([])
    plt.yticks([])
    # show
    plt.show()

def plot_model(model: torch.nn.Module):
    # concatenate all parameters into a single tensor
    params = torch.cat([p.data.view(-1) for p in model.parameters()])
    
    # plot
    plot_tensor(params)
    
def plot_model_diff(model1: torch.nn.Module, model2: torch.nn.Module):
    # concatenate all parameters into a single tensor
    params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
    
    # plot
    plot_tensor(params2 - params1)

    