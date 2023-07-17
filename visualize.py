import torch, math
import matplotlib.pyplot as plt

def make_square(t: torch.Tensor) -> torch.Tensor:
    t = t.view(-1)
    n = int(math.ceil(math.sqrt(t.numel())))
    t = torch.nn.functional.pad(t, (0, n**2-t.numel()))
    return t.view(n, n)

def plot_tensor(tensor: torch.Tensor, title: str = None):    
    # make square view
    tensor = make_square(tensor)

    # plot
    plt.figure(figsize=(10, 10))
    plt.imshow(tensor.cpu())
    # colorbar
    plt.colorbar()
    # no grid
    plt.xticks([])
    plt.yticks([])
    # title
    plt.title(title)
    # show
    plt.show()

def plot_model(model: torch.nn.Module, title: str = None):
    # concatenate all parameters into a single tensor
    params = torch.cat([p.data.view(-1) for p in model.parameters()])
    
    # plot
    plot_tensor(params)
    
def plot_model_diff(model1: torch.nn.Module, model2: torch.nn.Module, title: str = None):
    # concatenate all parameters into a single tensor
    params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
    params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
    
    # plot
    plot_tensor(params2 - params1, title)

def plot_model_params(
    model: torch.nn.Module,
    title: str = None,
    cols: int = 4,
    squared: bool = True,
):
    l = len(list(model.named_parameters()))
    rows = math.ceil(l / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    
    # plot each parameter
    for i, (name, params) in enumerate(model.named_parameters()):
        row, col = i // cols, i % cols
        ax = axs[row, col]
        if squared:
            params = make_square(params.data)
        else:
            params = params.data.view(params.size()[0], -1)
        ax.imshow(params)
        ax.set_title(name)
        
        # colorbar
        fig.colorbar(ax.images[0], ax=ax)
    
    # title
    if title:
        plt.suptitle(title)
        
    # show
    fig.tight_layout()
    plt.show()
    
def plot_model_params_diff(
    model0: torch.nn.Module,
    model1: torch.nn.Module,
    title: str = None,
    cols: int = 4,
    squared: bool = True,
):
    l = len(list(model0.named_parameters()))
    rows = math.ceil(l / cols)
    
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    
    # plot each parameter
    for i, ((name0, params0), (name1, params1)) in enumerate(
        zip(model0.named_parameters(), model1.named_parameters())
    ):
        if name0 != name1:
            raise ValueError(f"Parameter names do not match: {name0} != {name1}")
        
        row, col = i // cols, i % cols
        ax = axs[row, col]
        diff = params1 - params0
        if squared:
            diff = make_square(diff.data)
        else:
            diff = diff.data.view(diff.size()[0], -1)
        ax.imshow(diff)
        ax.set_title(name0)
    
        # colorbar
        fig.colorbar(ax.images[0], ax=ax)
    
    # title
    if title:
        plt.suptitle(title)
        
    # show
    fig.tight_layout()
    plt.show()
