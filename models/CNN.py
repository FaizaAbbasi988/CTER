# Convolution module
# use conv to capture local features, instead of postion embedding.
class CNNmod(nn.Module):
    def __init__(self, emb_size=1000):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 45), (1, 2)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 45), (1, 8)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            # nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),

        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class ClassificationHeadCNN(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(2200, 256),
            # nn.ELU(),
            # nn.Dropout(0.5),
        #     nn.Linear(256, 32),
        #     nn.ELU(),
        #     nn.Dropout(0.3),
        #     # nn.Linear(32, 4)
        )

    def forward(self, x):


        c_features = self.fc(x)
        return c_features, x
class CNN(nn.Sequential):
    def __init__(self, emb_size=1000, depth=6, n_classes=2, **kwargs):
        super().__init__(
            CNNmod(emb_size),
            ClassificationHeadCNN(emb_size, n_classes),
        )