import torch.nn as nn
from einops import rearrange


# input: b * ant * car * 2
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock_2D(nn.Module):
    def __init__(self, ant_size, car_size, dropout = 0.):
        super().__init__()
        self.ant_mix = nn.Sequential(
            nn.LayerNorm(ant_size*2),
            FeedForward(ant_size*2, ant_size*2*2, dropout),
        )
        self.car_mix = nn.Sequential(
            nn.LayerNorm(car_size*2),
            FeedForward(car_size*2, car_size*2*2, dropout),
        )

    def forward(self, x):
        x = rearrange(x, 'b ant car c -> b car (ant c)')
        x = x+self.ant_mix(x)
        x = rearrange(x, 'b car (ant c) -> b ant car c', c=2)

        x = rearrange(x, 'b ant car c -> b ant (car c)')
        x = x+self.car_mix(x)
        x = rearrange(x, 'b ant (car c) -> b ant car c', c=2)

        return x


class MLPMixer_us_2D(nn.Module):
    def __init__(self, ant_size, car_size, depth, dropout = 0.):
        super(MLPMixer_us_2D, self).__init__()
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock_2D(ant_size, car_size))

    def forward(self, x):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        return x


class Mapping_Net(nn.Module):
    def __init__(self, input_ant_size, input_car_size, ant_size, car_size, depth):
        super(Mapping_Net, self).__init__()
        self.fc1=nn.Linear(input_ant_size*2, ant_size*2)
        self.fc2=nn.Linear(input_car_size*2, car_size*2)
        self.mlpmixer_us = MLPMixer_us_2D(ant_size, car_size,depth)
        self.fc_reverse1 = nn.Linear(ant_size*2, ant_size*2)
        self.fc_reverse2 = nn.Linear(car_size*2, car_size*2)

    def forward(self, x):
        out = rearrange(x, 'b in_ant in_car c -> b in_car (c in_ant)')
        out = self.fc1(out)
        out = rearrange(out, 'b in_car (c ant) -> b in_car c ant', c=2)
        out = rearrange(out, 'b in_car c ant -> b ant (c in_car)')
        out = self.fc2(out)
        out = rearrange(out, 'b ant (c car) -> b ant car c', c=2)

        out = self.mlpmixer_us(out)

        out = rearrange(out, 'b ant car c -> b car (c ant)')
        out = self.fc_reverse1(out)
        out = rearrange(out, 'b car (c ant) -> b car c ant', c=2)
        out = rearrange(out, 'b car c ant -> b ant (c car)')
        out = self.fc_reverse2(out)
        out = rearrange(out, 'b ant (c car) -> b ant car c', c=2)
        return out
