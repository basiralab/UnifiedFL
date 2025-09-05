import math, torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, name=None, generator=None):
        super().__init__()
        layers, in_size = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.ReLU()]
            in_size = h
        layers += [nn.Linear(in_size, output_size)]
        self.net = nn.Sequential(*layers)
        self._initialize_weights(generator)
        self.name = name or f"{len(hidden_sizes)+1}-layer{'-'.join(str(h) for h in hidden_sizes)}"

    def _initialize_weights(self, generator=None):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity='relu', generator=generator)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound, generator=generator)

    def forward(self, x):
        return self.net(x.flatten(1))

    def __repr__(self): return self.name
    def __str__(self): return self.name
