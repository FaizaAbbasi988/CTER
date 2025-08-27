class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n h d -> b n h d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n h d -> b n h d", h=self.num_heads)
        values = rearrange(self.values(x), "b n h d -> b n h d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # out = rearrange(out, "b (h n) d -> b n (h d)")
        out = self.projection(out)
        return out
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        # x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=5,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),

                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
class ClassificationHeadTF(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.pools = 15
        input_size: int = 1000
        self.poolk = 75
        self.linear_len = (input_size-self.poolk)//self.pools+1
        self.fc = nn.Sequential(
            Rearrange('b e (h) (w) -> b (e h) w'),
            nn.AvgPool2d(kernel_size=(1,self.poolk),stride=(1,self.pools)),
            nn.Flatten(),
            # nn.Linear(1364, 256),
            # nn.ELU(),
            # nn.Dropout(0.5),
        #     nn.Linear(256,100),
        #     nn.ELU(),
        #     nn.Dropout(0.3),
        #     # nn.Linear(100, 4),
        )


    def forward(self, x):

        t_features = self.fc(x)
        return t_features,x