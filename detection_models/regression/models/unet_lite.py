import torch, torch.nn as nn

def conv(n_in, n_out):
    return nn.Sequential(
        nn.Conv2d(n_in,n_out,3,1,1), nn.BatchNorm2d(n_out), nn.ReLU(),
        nn.Conv2d(n_out,n_out,3,1,1), nn.BatchNorm2d(n_out), nn.ReLU()
    )

class UNetLite(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=16):
        super().__init__()
        self.e1, self.p1 = conv(in_ch,base), nn.MaxPool2d(2)
        self.e2, self.p2 = conv(base,base*2), nn.MaxPool2d(2)
        self.bott = conv(base*2,base*4)
        self.u2, self.d2 = nn.ConvTranspose2d(base*4,base*2,2,2), conv(base*4,base*2)
        self.u1, self.d1 = nn.ConvTranspose2d(base*2,base,2,2), conv(base*2,base)
        self.head = nn.Conv2d(base, out_ch, 1)
    def forward(self,x):
        e1=self.e1(x)
        e2=self.e2(self.p1(e1))
        b=self.bott(self.p2(e2))
        d2=self.d2(torch.cat([self.u2(b), e2],1))
        d1=self.d1(torch.cat([self.u1(d2), e1],1))
        return self.head(d1)
