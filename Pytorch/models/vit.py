import torch
from torch import nn

from models.layers import Lambda
from models.embeddings import PatchEmbdding, CLSToken, AbsPosEmbedding
from models.attentions import Transformer


class ViT(nn.Module):
    def __init__(self, spectra_size, patch_size, num_classes, dim, depth, heads, dim_mlp, channel=1, dim_head=16, dropout=0.0, emb_dropout=0.0, sd=0.0, embedding=None, classifier=None, name='vit', **block_kwargs):
        super(ViT, self).__init__()
        self.name = name
        self.embedding =nn.Sequential(
            PatchEmbdding(spectra_size=spectra_size, patch_size=patch_size, dim_out=dim, channel=channel),
            CLSToken(dim=dim),
            AbsPosEmbedding(spectra_size=spectra_size, patch_size=patch_size, dim=dim, cls=True),
            nn.Dropout(emb_dropout) if emb_dropout > 0.0 else nn.Identity(),
        )if embedding is None else embedding
    
        self.transformers = []
        for i in range(depth):
            self.transformers.append(
                Transformer(dim, heads=heads, dim_head=dim_head, dim_mlp=dim_mlp, dropout=dropout, sd=(sd * i / (depth -1)))
            )
        self.transformers = nn.Sequential(*self.transformers)


        
        self.classifier = nn.Sequential(
            # Lambda(lambda x: x[:, 0]),
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )if classifier is None else classifier
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformers(x)
        x = self.classifier(x[:, 0])
        
        return x

