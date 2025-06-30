import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from torch.nn.attention import SDPBackend, sdpa_kernel
import math
from einops import rearrange
from einops.layers.torch import Rearrange

import copy
from functools import partial

class GLU(nn.Module):
    def forward(self, x, gates=None):
        if gates is None:
            x, gates = x.chunk(2, dim=-1)
        return self.act(x) * gates

class GeGLU(GLU):
    def __init__(self):
        super().__init__()
        self.act = nn.GELU()


class SwiGLU(GLU):
    def __init__(self):
        super().__init__()
        self.act = nn.SiLU()


def get_activation(act="gelu"):
    match act:
        case "relu":
            return nn.ReLU
        case "gelu":
            return nn.GELU
            # return partial(nn.GELU, approximate="tanh")
        case "silu":
            return nn.SiLU
        case "tanh":
            return nn.Tanh
        case "geglu":
            return GeGLU
        case "swiglu":
            return SwiGLU
        case _:
            raise ValueError(f"Unknown activation function: {act}")

def get_embeddings(size, type=None):
    match type:
        case None:
            patch_embeddings = nn.Parameter(torch.randn(*size))
        case "normalize":
            dim = size[-1]
            patch_embeddings = nn.Parameter((dim**-0.5) * torch.randn(*size))
        case "bert":
            patch_embeddings = nn.Parameter(torch.empty(*size).normal_(std=0.02))
        case _:
            raise ValueError(f"Unknown type for embedding: {type}")
    return patch_embeddings

def layer_initialize(layer, mode="zero", gamma=0.01):
    # re-initialize given layer to have small outputs
    if mode == "zero":
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif mode == "uniform":
        nn.init.uniform_(layer.weight, -gamma, gamma)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -gamma, gamma)
    else:
        raise ValueError(f"Unknown mode {mode}")



class ConvEmbedder(nn.Module):
    """
    Preprocess data (break into patches) and embed them into target dimension.
    """
    def __init__(self, config, x_num, data_dim):
        super().__init__()
        self.config = config

        self.dim = config["dim"]
        self.data_dim = data_dim
        act = get_activation("gelu")

        assert (
            x_num % config["patch_num"] == 0
        ), f"x_num must be divisible by patch_num, x_num: {x_num}, patch_num: {config['patch_num']}"
        self.patch_resolution = x_num // config["patch_num"]  # resolution of one space dimension for each patch
        self.patch_dim = data_dim * self.patch_resolution * self.patch_resolution  # dimension per patch

        assert (
            x_num % config["patch_num_output"] == 0
        ), f"x_num must be divisible by patch_num_output, x_num: {x_num}, patch_num_output: {config['patch_num_output']}"
        self.patch_resolution_output = (
            x_num // config["patch_num_output"]
        )  # resolution of one space dimension for each patch in output
        self.patch_dim_output = (
            data_dim * self.patch_resolution_output * self.patch_resolution_output
        )  # dimension per patch in output

        ## for encoder part

        self.patch_position_embeddings = get_embeddings((1, 1, config["patch_num"] * config["patch_num"], self.dim))

        self.time_embed_type = config.get("time_embed", "continuous")
        match self.time_embed_type:
            case "continuous":
                self.time_proj = nn.Sequential(
                    nn.Linear(1, self.dim),
                    act(),
                    nn.Linear(self.dim, self.dim),
                )
            case "learnable":
                self.time_embeddings = get_embeddings((1, config.get("max_time_len", 10), 1, self.dim))

        if config.get("early_conv", 0):
            n_conv_layers = math.log2(self.patch_resolution)
            assert n_conv_layers.is_integer(), f"patch_resolution {self.patch_resolution} must be a power of 2"
            n_conv_layers = int(n_conv_layers)
            kernel_size = [3] * n_conv_layers + [1]
            stride = [2] * n_conv_layers + [1]
            padding = [1] * n_conv_layers + [0]
            channels = [data_dim] + [self.dim // (2**i) for i in range(n_conv_layers - 1, 0, -1)] + [self.dim, self.dim]

            self.conv_proj = nn.Sequential()
            for i in range(len(kernel_size)):
                self.conv_proj.append(
                    nn.Conv2d(
                        in_channels=channels[i],
                        out_channels=channels[i + 1],
                        kernel_size=kernel_size[i],
                        stride=stride[i],
                        padding=padding[i],
                    )
                )
                if i < len(kernel_size) - 1:
                    self.conv_proj.append(act())
        else:
            # regular vit patch embedding
            self.in_proj = nn.Conv2d(
                in_channels=data_dim,
                out_channels=self.dim,
                kernel_size=self.patch_resolution,
                stride=self.patch_resolution,
            )
            self.conv_proj = nn.Sequential(
                act(),
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim, kernel_size=1, stride=1),
            )

        ## for decoder part

        self.conv_dim = config.get("conv_dim", self.dim // 4)

        if config.get("deep", 0):
            self.post_proj = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=self.dim),
                act(),
                nn.Linear(in_features=self.dim, out_features=self.dim),
                act(),
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config["patch_num_output"], w=self.config["patch_num_output"]),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
            )
        else:
            self.post_proj = nn.Sequential(
                Rearrange("b (t h w) d -> (b t) d h w", h=self.config["patch_num_output"], w=self.config["patch_num_output"]),
                nn.ConvTranspose2d(
                    in_channels=self.dim,
                    out_channels=self.conv_dim,
                    kernel_size=self.patch_resolution_output,
                    stride=self.patch_resolution_output,
                ),
                act(),
                nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim, kernel_size=1, stride=1),
                act(),
            )
        self.head = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.data_dim, kernel_size=1, stride=1)

        if config.get("initialize_small_output", 0):
            layer_initialize(self.head, mode=config["initialize_small_output"])

    def encode(self, data, skip_len=0):
        """
        Input:
            data:           Tensor (bs, input_len, x_num, x_num, data_dim)
            times:          Tensor (bs, input_len, 1)
        Output:
            data:           Tensor (bs, data_len, dim)      data_len = input_len * patch_num * patch_num
                            embedded data + time embeddings + patch position embeddings
        """

        bs = data.size(0)
        data = rearrange(data[:, skip_len:], "b t h w c -> (b t) c h w")
        data = self.in_proj(data)
        data = self.conv_proj(data)  # (bs*input_len, d, patch_num, patch_num)
        data = rearrange(data, "(b t) d h w -> b t (h w) d", b=bs)  # (bs, input_len, p*p, dim)

        data = data + self.patch_position_embeddings  # (b, input_len, p*p, d)

        data = data.reshape(bs, -1, self.dim)
        return data

    def decode(self, data_output):
        """
        Input:
            data_output:     Tensor (bs, query_len, dim)
                             query_len = output_len * patch_num * patch_num
        Output:
            data_output:     Tensor (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_output.size(0)
        data_output = self.post_proj(data_output)  # (bs*output_len, data_dim, x_num, x_num)
        data_output = self.head(data_output)
        data_output = rearrange(data_output, "(b t) c h w -> b t h w c", b=bs)
        return data_output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_block_attn_mask(block_size: int, n_repeat: int, device=torch.device("cpu")):
    """
    Output:
        attn_mask: BoolTensor (block_size * n_repeat, block_size * n_repeat) block diagonal matrix with identity blocks
    """
    blocks = [torch.ones(block_size, block_size, device=device)] * n_repeat
    return torch.block_diag(*blocks).bool()


def block_lower_triangular_mask(block_size, block_num, use_float=False):
    """
    Create a block lower triangular boolean mask. (upper right part will be 1s, and represent locations to ignore.)
    """
    matrix_size = block_size * block_num
    lower_tri_mask = torch.tril(torch.ones(matrix_size, matrix_size, dtype=torch.bool))
    block = torch.ones(block_size, block_size, dtype=torch.bool)
    blocks = torch.block_diag(*[block for _ in range(block_num)])
    final_mask = torch.logical_or(lower_tri_mask, blocks)

    if use_float:
        return torch.zeros_like(final_mask, dtype=torch.float32).masked_fill_(~final_mask, float("-inf"))
    else:
        return ~final_mask


def block_causal(b, h, q_idx, kv_idx, block_size):
    return (q_idx // block_size) >= (kv_idx // block_size)


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, act="gelu", dropout=0):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)

        if act.endswith("glu"):
            self.fc_gate = nn.Linear(dim, hidden_dim)
        else:
            self.fc_gate = None

        self.activation = get_activation(act)()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        if self.fc_gate is None:
            return self.fc2(self.dropout(self.activation(self.fc1(x))))
        else:
            return self.fc2(self.dropout(self.activation(self.fc1(x), self.fc_gate(x))))


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, qk_norm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.linear_q = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_k = nn.Linear(embed_dim, embed_dim, bias)
        self.linear_v = nn.Linear(embed_dim, embed_dim, bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias)

        self.qk_norm = qk_norm
        if qk_norm:
            # self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
            # self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-5)
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-5)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        block_mask=None,
        is_causal=False,
        rotary_emb=None,
        cache=None,
    ):
        bs, seq_len, _ = query.size()
        k_len = key.size(1)

        # compute projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # split heads (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # if rotary_emb is not None:
        #     q = rotary_emb(q)
        #     k = rotary_emb(k)

        # (bs, n_head, seq_len, head_dim)
        # q = q.transpose(1, 2)
        q = q.transpose(1, 2).contiguous()  # make torch.compile happy, striding error otherwise
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rotary_emb is not None:
            q, k = rotary_emb.rotate_queries_with_cached_keys(q, k)

        if cache is not None:
            k, v = cache.update(k, v)
            k_len = k.size(2)

        # process and merge masks
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                bs,
                k_len,
            ), f"expecting key_padding_mask shape of {(bs, k_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bs, 1, 1, k_len).expand(-1, self.num_heads, -1, -1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        dropout_p = 0.0 if not self.training else self.dropout

        # with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask, dropout_p, is_causal
        )  # (bs, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output)


class MultiheadFlexAttention(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, qk_norm=False):
        super().__init__(embed_dim, num_heads, dropout, bias, qk_norm)
        # self.flex_sdpa = torch.compile(flex_attention, dynamic=False)
        self.flex_sdpa = torch.compile(flex_attention)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        block_mask=None,
        is_causal=False,
        rotary_emb=None,
        cache=None,
    ):

        bs, seq_len, _ = query.size()
        k_len = key.size(1)

        # compute projections
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        # split heads (bs, seq_len, dim) -> (bs, n_heads, seq_len, head_dim)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim)
        k = k.view(bs, k_len, self.num_heads, self.head_dim)
        v = v.view(bs, k_len, self.num_heads, self.head_dim)

        if self.qk_norm:
            dtype = q.dtype  # it seems flexattention doesn't autocast to bfloat16
            q = self.q_norm(q).to(dtype)
            k = self.k_norm(k).to(dtype)

        # (bs, n_head, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if rotary_emb is not None:
            q, k = rotary_emb.rotate_queries_with_cached_keys(q, k)

        if cache is not None:
            k, v = cache.update(k, v)
            k_len = k.size(2)

        output = self.flex_sdpa(q, k, v, block_mask=block_mask)  # (bs, n_heads, seq_len, head_dim)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.out_proj(output)


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    Custom implementation of pytorch's TransformerEncoderLayer
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0,
        attn_dropout: float = 0,
        activation = F.relu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        rotary=False,
        norm=nn.LayerNorm,
        qk_norm=False,
        flex_attn=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(nn.TransformerEncoderLayer, self).__init__()

        if flex_attn:
            self.self_attn = MultiheadFlexAttention(d_model, nhead, dropout=attn_dropout, bias=bias, qk_norm=qk_norm)
        else:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=attn_dropout, bias=bias, qk_norm=qk_norm)
        self.rotary = rotary

        self.ffn = FFN(d_model, dim_feedforward, activation, dropout)

        self.norm_first = norm_first

        self.norm1 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = norm(d_model, eps=layer_norm_eps, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        src,
        src_mask = None,
        src_key_padding_mask = None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                block_mask=block_mask,
                is_causal=is_causal,
                rotary_emb=rotary_emb,
            )
            x = x + self.dropout2(self.ffn(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x, src_mask, src_key_padding_mask, block_mask=block_mask, is_causal=is_causal, rotary_emb=rotary_emb
                )
            )
            x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

    def _sa_block(
        self,
        x,
        attn_mask = None,
        key_padding_mask= None,
        block_mask=None,
        is_causal: bool = False,
        rotary_emb=None,
    ):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            block_mask=block_mask,
            is_causal=is_causal,
            rotary_emb=rotary_emb,
        )
        return self.dropout1(x)


class CustomTransformerEncoder(nn.Module):
    """
    Custom implementation of pytorch's TransformerEncoder
    Source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
    """

    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        norm = None,
        config=None,
    ) -> None:
        super().__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        self.rotary_emb = None
        self.rotary = False

    def forward(self, src, mask=None, src_key_padding_mask=None, block_mask=None, is_causal = False):
        # prepare masks
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        for mod in self.layers:
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                block_mask=block_mask,
                rotary_emb=self.rotary_emb,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class BCAT(nn.Module):
    """
    Wrapper for the autoregressive BCAT model.
    """

    def __init__(self, config, x_num, max_output_dim, max_data_len=1):
        super().__init__()
        self.config = config
        self.x_num = x_num
        self.max_output_dim = max_output_dim

        self.embedder = ConvEmbedder(config["embedder"], x_num, max_output_dim)
        self.flex_attn = config.get("flex_attn", False)

        match config.get("norm", "layer"):
            case "rms":
                norm = nn.RMSNorm
            case _:
                norm = nn.LayerNorm

        kwargs = {
            "d_model": config["dim_emb"],
            "nhead": config["n_head"],
            "dim_feedforward": config["dim_ffn"],
            "dropout": config["dropout"],
            "attn_dropout": config.get("attn_dropout", 0),
            "activation": config.get("activation", "gelu"),
            "norm_first": config["norm_first"],
            "norm": norm,
            "rotary": config["rotary"],
            "qk_norm": config.get("qk_norm", False),
            "flex_attn": self.flex_attn,
        }

        self.transformer = CustomTransformerEncoder(
            CustomTransformerEncoderLayer(**kwargs),
            num_layers=config["n_layer"],
            norm=norm(config["dim_emb"], eps=1e-5) if config["norm_first"] else None,
            config=config,
        )

        self.seq_len_per_step = config["embedder"]["patch_num"]**2
        mask = block_lower_triangular_mask(self.seq_len_per_step, max_data_len, use_float=True)
        self.register_buffer("mask", mask, persistent=False)

        if self.flex_attn:
            block_size = config["patch_num"]**2
            seq_len = block_size * (max_data_len - 1)
            self.block_mask = create_block_mask(
                partial(block_causal, block_size=block_size), None, None, seq_len, seq_len
            )
            self.block_size = block_size
            self.block_mask_prefil = None

    def summary(self):
        s = "\n"
        s += f"\tEmbedder:        {sum([p.numel() for p in self.embedder.parameters() if p.requires_grad]):,}\n"
        s += f"\tTransformer:    {sum([p.numel() for p in self.transformer.parameters() if p.requires_grad]):,}"
        return s

    def forward(self, x:torch.Tensor):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        #if mode == "fwd":
        x = x.permute([0, 2, 3, 1]).unsqueeze(1).contiguous()
        return self.fwd(x, 1)
        #elif mode == "generate":
        #    return self.generate(**kwargs)
        #else:
        #    raise Exception(f"Unknown mode: {mode}")

    def fwd(self, data, input_len: int, **kwargs):
        """
        Inputs:
            data:          Tensor     (bs, input_len+output_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            input_len:     How many timesteps to use as input, for training this should be 1

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        data = data[:, :]  # ignore last timestep for autoregressive training (b, t_num-1, x_num, x_num, data_dim)

        """
        Step 1: Prepare data input (add time embeddings and patch position embeddings)
            data_input (bs, t_num-1, x_num, x_num, data_dim) -> (bs, data_len, dim)
                       data_len = (input_len + output_len - 1) * patch_num * patch_num
        """

        data = self.embedder.encode(data)  # (bs, data_len, dim)
        """
        Step 2: Transformer
            data_input:   Tensor     (bs, data_len, dim)
        """
        data_len = data.size(1)
        if self.flex_attn:
            block_mask = self.block_mask
            data_encoded = self.transformer(data, block_mask=block_mask)  # (bs, data_len, dim)
        else:
            mask = self.mask[:data_len, :data_len]
            #with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            data_encoded = self.transformer(data, mask=mask)  # (bs, data_len, dim)

        """
        Step 3: Decode data
        """

        input_seq_len = (input_len - 1) * self.seq_len_per_step
        data_output = data_encoded[:, input_seq_len:]  # (bs, output_len*patch_num*patch_num, dim)

        data_output = self.embedder.decode(data_output).squeeze(1).permute(0,3, 1, 2)  # (bs, output_len, x_num, x_num, data_dim)
        return data_output

    @torch.compiler.disable()
    def generate(self, data_input, times, input_len: int, data_mask, carry_over_c=-1, **kwargs):
        """
        Inputs:
            data_input:    Tensor     (bs, input_len, x_num, x_num, data_dim)
            times:         Tensor     (bs/1, input_len+output_len, 1)
            data_mask:     Tensor     (1, 1, 1, 1, data_dim)
            carry_over_c:  int        Indicate channel that should be carried over,
                                        not masked out or from output (e.g. boundary mask channel)

        Output:
            data_output:     Tensor     (bs, output_len, x_num, x_num, data_dim)
        """

        t_num = times.size(1)
        output_len = t_num - input_len
        bs, _, x_num, _, data_dim = data_input.size()

        data_all = torch.zeros(bs, t_num, x_num, x_num, data_dim, dtype=data_input.dtype, device=data_input.device)
        data_all[:, :input_len] = data_input
        cur_len = input_len
        prev_len = 0

        config = self.config

        if self.flex_attn and self.block_mask_prefil is None:
            seq_len_eval = self.block_size * input_len
            self.block_mask_prefil = create_block_mask(
                partial(block_causal, block_size=self.block_size), None, None, seq_len_eval, seq_len_eval
            )

        for i in range(output_len):
            cur_data_input = data_all[:, :cur_len]  # (bs, cur_len, x_num, x_num, data_dim)

            # (bs, cur_len, x_num, x_num, data_dim) -> (bs, data_len=cur_len*p*p, dim)
            skip_len = prev_len if self.config["kv_cache"] else 0
            cur_data_input = self.embedder.encode(
                cur_data_input, times[:, :cur_len], skip_len=skip_len
            )  # (bs, data_len, dim)

            mask = block_mask = None
            if (not self.config["kv_cache"]) or i == 0:
                if self.flex_attn:
                    block_mask = self.block_mask_prefil
                else:
                    data_len = cur_len * self.seq_len_per_step
                    mask = self.mask[:data_len, :data_len]

            cur_data_encoded = self.transformer(cur_data_input, mask, block_mask=block_mask)  # (bs, data_len, dim)

            new_output = cur_data_encoded[:, -self.seq_len_per_step :]  # (bs, patch_num*patch_num, dim)
            new_output = self.embedder.decode(new_output)  # (bs, 1, x_num, x_num, data_dim)

            new_output = new_output * data_mask  # (bs, 1, x_num, x_num, data_dim)

            if carry_over_c >= 0:
                new_output[:, 0, :, :, carry_over_c] = data_all[:, 0, :, :, carry_over_c]

            data_all[:, cur_len : cur_len + 1] = new_output
            prev_len = cur_len
            cur_len += 1

        return data_all[:, input_len:]
