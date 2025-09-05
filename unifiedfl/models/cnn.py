import math, torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, name=None, generator=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = (3, 3)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=self.kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*28*28, num_classes)
        )
        self._initialize_weights(generator)
        if name is None:
            info = []
            for l in self.net:
                if isinstance(l, nn.Conv2d):
                    info.append(f"Conv({l.in_channels}->{l.out_channels}, {l.kernel_size[0]}x{l.kernel_size[1]})")
                elif isinstance(l, nn.Linear):
                    info.append(f"Linear({l.in_features}->{l.out_features})")
            self.name = f"SimpleCNN-{len(info)}-layers: " + " -> ".join(info)
        else:
            self.name = name

    def _initialize_weights(self, generator=None):
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                n = self.in_channels
                for k in self.kernel_size: n *= k
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv, generator=generator)
                if m.bias is not None: m.bias.data.uniform_(-stdv, stdv, generator=generator)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity='relu', generator=generator)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound, generator=generator)

    def forward(self, x): return self.net(x)
    def __repr__(self): return self.name
    def __str__(self): return self.name

class CNNClassifierDeep(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, name=None, generator=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = (3, 3)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=self.kernel_size, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*4*8, num_classes)
        )
        self._initialize_weights(generator)
        if name is None:
            info = []
            for l in self.net:
                if isinstance(l, nn.Conv2d):
                    info.append(f"Conv({l.in_channels}->{l.out_channels}, {l.kernel_size[0]}x{l.kernel_size[1]})")
                elif isinstance(l, nn.Linear):
                    info.append(f"Linear({l.in_features}->{l.out_features})")
            self.name = f"SimpleCNN-{len(info)}-layers: " + " -> ".join(info)
        else:
            self.name = name

    def _initialize_weights(self, generator=None):
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                n = self.in_channels
                for k in self.kernel_size: n *= k
                stdv = 1. / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv, generator=generator)
                if m.bias is not None: m.bias.data.uniform_(-stdv, stdv, generator=generator)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity='relu', generator=generator)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound, generator=generator)

    def forward(self, x): return self.net(x)
    def __repr__(self): return self.name
    def __str__(self): return self.name
