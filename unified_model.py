from models.CNN import CNN, ClassificationHeadCNN, CNNmod
from models.transformer import TransformerEncoder, TransformerEncoderBlock, ClassificationHeadTF

class CNN(nn.Sequential):
    def __init__(self, emb_size=1000, depth=6, n_classes=4, **kwargs):
        super().__init__(
            CNNmod(emb_size),
            ClassificationHeadCNN(emb_size, n_classes)
        )
class Transformer(nn.Sequential):
    def __init__(self, emb_size=1000, depth=6, n_classes=4, **kwargs):
        super().__init__(
            TransformerEncoder(depth, emb_size),
            ClassificationHeadTF(emb_size, n_classes)
        )
class ConcatenatedClassifier(nn.Module):
    def __init__(self, c_features, t_features, n_classes):
        super().__init__()
        # total_feature_size = c-features + t-features
        # print("total_feature_size output shape:", total_feature_size)
        self.classifier = nn.Sequential(
            nn.Linear(3564, 100),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(100, n_classes)
        )

    def forward(self, c_features, t_features):
        # c_features = torch.tensor(c_features)
        xx = torch.cat((t_features[0], c_features[0]), dim=1)
        last_100_features = xx[:,-512:].detach().numpy()
        np.save('filename.npy', last_100_features)
        return self.classifier(xx)

# Unified model
class UnifiedModel(nn.Module):
    def __init__(self, cnn_emb_size = 1000, transformer_emb_size = 1000, n_classes= 4, **kwargs):
        super().__init__()
        self.cnn = CNN(emb_size=1000)
        self.transformer = Transformer(emb_size=1000)
        self.classifier = ConcatenatedClassifier(1000,1000, n_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        out = self.classifier(cnn_features, transformer_features)
        return x, out