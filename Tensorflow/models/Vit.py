import tensorflow as tf
from tensorflow import keras
from keras import Model, layers, initializers
import numpy as np


'''
    The reflection spectral dimension is (1400, 1), Patch_size 140, embed_dim 512, number of head 16, Depth 8, Dropout 0.0
    The transmission spectral dimension is (960, 1), Patch_size 120, embed_dim 512, number of head 16, Depth 8, Dropout 0.1
'''

class PatchEmbed(layers.Layer):
    # 1D Spectra to Patch Embedding
    def __init__(self, spectra_size=1400, patch_size=140, embed_dim=512):
        super(PatchEmbed, self).__init__()
        self.spectra_size = spectra_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        if self.spectra_size % self.patch_size != 0:            # When integer segmentation is not possible, the spectrum is filled
            self.spectra_size = self.spectra_size + (self.patch_size - self.spectra_size % self.patch_size) % self.patch_size
        self.num_patches = self.spectra_size // patch_size
        self.proj = layers.Conv1D(filters=embed_dim, kernel_size=patch_size, strides=patch_size, padding='SAME', kernel_initializer=initializers.LecunNormal(), bias_initializer=initializers.Zeros())

    def call(self, inputs, **kwargs):
        B, H, C = inputs.shape
        # assert H == self.spectra_size, f"Input Spectra size ({H}) doesn't match model ({self.spectra_size})."
        # 对分块不能整除进行填充
        pad_r = 0
        if H != self.spectra_size:
            pad_r = (self.patch_size - H % self.patch_size) % self.patch_size
            inputs = tf.pad(inputs, paddings=[[0, 0], [0, pad_r], [0, 0]])
        x = self.proj(inputs)       # (None, 10, 512)
        return x


class ConcatClassTokenAddPosEmbed(layers.Layer):
    def __init__(self, embed_dim=512, num_patches=10, name=None):
        super(ConcatClassTokenAddPosEmbed, self).__init__(name=name)
        self.embed_dim = embed_dim
        self.num_patches= num_patches

    def build(self, input_shape):
        self.cls_token = self.add_weight(name='cls', shape=[1, 1, self.embed_dim], initializer=initializers.Zeros(), trainable=True, dtype=tf.float32)
        self.pos_embed = self.add_weight(name='pos_embed', shape=[1, self.num_patches+1, self.embed_dim], initializer=initializers.RandomNormal(stddev=0.02), trainable=True, dtype=tf.float32)
        super(ConcatClassTokenAddPosEmbed, self).build(input_shape)

    def call(self, inputs, **kwargs):
        batch_size, _, _ = inputs.shape     # (10, 512)
        cls_token = tf.broadcast_to(self.cls_token, shape=[batch_size, 1, self.embed_dim])
        x = tf.concat([cls_token, inputs], axis=1)
        x = x + self.pos_embed                # 增加位置编码
        return x


class Attention(layers.Layer):
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.Zeros()

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., name=None):
        super(Attention, self).__init__(name=name)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name='qkv', kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.attn_drop = layers.Dropout(attn_drop_ratio)
        self.proj = layers.Dense(dim, name='out', kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.proj_drop = layers.Dropout(proj_drop_ratio)

    def call(self, inputs, training=None):
        B, N, C = inputs.shape      # (1, 341, 512)
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, [B, N, 3, self.num_heads, C // self.num_heads])       # (1, 10, 3, 16, 32)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])        # (3, 1, 16, 10, 32)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(a=q, b=k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    
class MLP(layers.Layer):
    # MLP as used in Vision Transformer, MLP-Mixer and related networks
    k_ini = initializers.GlorotUniform()
    b_ini = initializers.RandomNormal(stddev=1e-6)

    def __init__(self, in_features, mlp_ratio=2.0, drop=0., name=None):
        super(MLP, self).__init__(name=name)
        self.fc1 = layers.Dense(int(in_features * mlp_ratio), name='Dense_0', kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.act = layers.Activation('gelu')
        # self.act = layers.Activation('relu')
        # self.act = layers.Activation('tanh')
        self.fc2 = layers.Dense(in_features, name='Dense_1', kernel_initializer=self.k_ini, bias_initializer=self.b_ini)
        self.drop = layers.Dropout(drop)

    def call(self, inputs, training=None):
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x


class Block(layers.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., name=None):
        super(Block, self).__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6, name='LayerNorm_0')
        # self.norm1 = layers.BatchNormalization(epsilon=1e-6, name='BatchNorm_0')
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio, name='MultiHeadAttention')
        self.drop_path = layers.Dropout(rate=drop_path_ratio, noise_shape=(None, 1, 1)) if drop_path_ratio > 0. else layers.Activation('linear')
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name='LayerNorm_1')
        # self.norm2 = layers.BatchNormalization(epsilon=1e-6, name='BatchNorm_1')
        self.mlp = MLP(dim, drop=drop_ratio, name='MLpBlock')

    def call(self, inputs, training=None):
        x = inputs + self.drop_path(self.attn(self.norm1(inputs)), training=training)
        x = x + self.drop_path(self.mlp(self.norm2(x)), training=training)
        return x


class VisionTransformer(Model):
    def __init__(self, spectra_size=1400, embed_dim=512, patch_size=140, depth=8, num_heads=16, qkv_bias=True, qk_scale=None, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., representation_size=None, num_classes=40, name='ViT'):
        super(VisionTransformer, self).__init__(name=name)
        self.spectra_size = spectra_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_head = num_heads
        self.qkv_bias = qkv_bias

        self.patch_embed = PatchEmbed(spectra_size=spectra_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token_pos_embed = ConcatClassTokenAddPosEmbed(embed_dim=embed_dim, num_patches=num_patches, name='cls_pos')
        self.pos_drop = layers.Dropout(drop_ratio)

        dpr = np.linspace(0., drop_path_ratio, depth)       # stochastic depth decay rule
        self.blocks = [Block(dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], name='encoderblock_{}'.format(i)) for i in range(depth)]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name='encoder_norm')
        # self.norm = layers.BatchNormalization(epsilon=1e-6, name='encoder_norm')

        if representation_size:
            self.has_logits = True
            self.pre_logits = layers.Dense(representation_size, activation='tanh', name='pre_logits')
        else:
            self.has_logits = False
            self.pre_logits = layers.Activation('linear')
        self.flatten = layers.Flatten()
        self.head = layers.Dense(num_classes, activation='softmax', name='head', kernel_initializer=initializers.Zeros())

    def call(self, inputs, training=None):
        x = self.patch_embed(inputs)
        x = self.cls_token_pos_embed(x)
        x = self.pos_drop(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)
        
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        # x = self.flatten(x)
        x = self.head(x)

        return x


def vit_base(num_classes: int=192, has_logits: bool=True):
    model = VisionTransformer(spectra_size=960, embed_dim=512, patch_size=120, depth=8, num_heads=16, representation_size=960 if has_logits else None, num_classes=num_classes, name='ViT', drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.)
    # model = VisionTransformer(spectra_size=1400, embed_dim=512, patch_size=140, depth=8, num_heads=16, representation_size=1400 if has_logits else None, num_classes=num_classes, name='ViT', drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.)
    return model

