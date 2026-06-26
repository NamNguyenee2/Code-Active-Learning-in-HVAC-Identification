import torch
import gpytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def train_GP(X_tr, y_tr, save_path='checkpoints/model_AL_Fisher_GP.pth', ori = 0, epochs=5):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(X_tr, y_tr, likelihood).to(device)
    if ori == 1:
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Rprop(model.parameters(), lr=1e-2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(epochs):
        optimizer.zero_grad()
        output = model(X_tr)
        loss = -mll(output, y_tr)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), save_path)

def evaluate_gp(X_tr, y_tr, X_t, y_t, save_path='checkpoints/model_AL_Fisher_GP.pth'):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(X_tr, y_tr, likelihood).to(device)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_t))
        mean = observed_pred.mean
        rmse = torch.sqrt(torch.mean((mean - y_t)**2))
    return rmse

def call_model(X_tr, y_tr,save_path):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = ExactGPModel(X_tr, y_tr, likelihood).to(device)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict)
    return model, likelihood

def cal_pos(X_tr, y_tr, X_t, save_path):
    model, likelihood = call_model(X_tr, y_tr,save_path)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_t))
        mean = observed_pred.mean
        var = observed_pred.variance
    return mean, var

