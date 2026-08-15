"""Twice-differentiable network backbones for the SURF PINN study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class NetworkSpec:
    architecture: Literal["resnet", "mlp"] = "resnet"
    width: int = 50
    blocks: int = 4
    layers_per_block: int = 2
    hidden_layers: int = 6

    def __post_init__(self) -> None:
        if self.architecture not in {"resnet", "mlp"}:
            raise ValueError("architecture must be 'resnet' or 'mlp'.")
        if min(self.width, self.blocks, self.layers_per_block, self.hidden_layers) < 1:
            raise ValueError("network dimensions must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_network(spec: NetworkSpec) -> Any:
    torch = _torch()

    class ResidualBlock(torch.nn.Module):
        def __init__(self, width: int, layers: int) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(width, width) for _ in range(layers)]
            )

        def forward(self, values: Any) -> Any:
            transformed = values
            for index, layer in enumerate(self.layers):
                transformed = layer(transformed)
                if index < len(self.layers) - 1:
                    transformed = torch.tanh(transformed)
            return values + transformed

    class ResNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = torch.nn.Linear(2, spec.width)
            self.blocks = torch.nn.ModuleList(
                [ResidualBlock(spec.width, spec.layers_per_block) for _ in range(spec.blocks)]
            )
            self.output = torch.nn.Linear(spec.width, 1)

        def forward(self, coordinates: Any) -> Any:
            values = torch.tanh(self.input(coordinates))
            for block in self.blocks:
                values = block(values)
            return self.output(values)

    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = [torch.nn.Linear(2, spec.width), torch.nn.Tanh()]
            for _ in range(spec.hidden_layers - 1):
                layers.extend((torch.nn.Linear(spec.width, spec.width), torch.nn.Tanh()))
            layers.append(torch.nn.Linear(spec.width, 1))
            self.network = torch.nn.Sequential(*layers)

        def forward(self, coordinates: Any) -> Any:
            return self.network(coordinates)

    model = ResNet() if spec.architecture == "resnet" else MLP()
    _initialize(model)
    return model


def count_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _initialize(model: Any) -> None:
    torch = _torch()
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.zeros_(module.bias)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for PINN networks.") from exc
    return torch
