from sklearn.base import BaseEstimator, RegressorMixin
import torch
import torch.nn as nn
import torch.optim as optim


class TorchExpModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.T = nn.Parameter(torch.randn(output_dim, input_dim) * 0.01)
        self.c = nn.Parameter(torch.zeros(output_dim))
        self.k = nn.Parameter(torch.ones(output_dim))

    def forward(self, x):
        return (
            self.c + self.k * torch.exp(self.T @ x.T).T
        )  # shape: (batch_size, output_dim)


class TorchExpEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim,
        output_dim,
        lr=0.1,
        epochs=5000,
        weight_decay=1e-3,
        delta=0.03,
        device="cpu",
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.delta = delta
        self.device = device

        self._build_model()

    def _build_model(self):
        self.model = TorchExpModel(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.loss_fn = lambda pred, target: nn.functional.huber_loss(
            pred, target, delta=self.delta
        )

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % (self.epochs // 5) == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            return self.model(X).cpu().numpy()
