import torch
import torch.nn as nn
import torch.optim as optim
import math

class ImagePatchEmbedding(nn.Module):
    """
    Converts a 2D image into a sequence of flattened patch embeddings.

    Args:
        img_size (int): The size (height and width) of the input image.
        patch_size (int): The size of each square patch.
        in_channels (int): The number of input channels.
        d_model (int): The dimension of the output patch embeddings.
    """
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super(ImagePatchEmbedding, self).__init__()
        # Ensure image dimensions are divisible by patch size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        # Calculate the number of patches along one dimension (e.g., width or height)
        self.num_patches_per_dim = img_size // patch_size
        # Calculate the total number of patches (sequence length for the Transformer)
        self.num_patches = self.num_patches_per_dim * self.num_patches_per_dim

        # Calculate the size of a flattened patch
        self.patch_dim = in_channels * (patch_size ** 2)

        # Convolutional layer to extract patch features
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size) # Change to KANSCONV?

    def forward(self, x):
        # Apply convolution to create patch embeddings
        x = self.proj(x)

        # Flatten the spatial dimensions into a sequence dimension
        x = x.flatten(2)

        # Transpose to get the desired Transformer input shape: (batch_size, num_patches, d_model)
        x = x.transpose(1, 2)
        return x
    

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.

    Args:
        d_model (int): The total dimension of the model.
        num_heads (int): The number of parallel attention heads. d_model must be
                         divisible by num_heads.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForward(nn.Module):
    """
    Implements a two-layer feed-forward network (MLP).

    This network is applied to each position independently. It consists of a
    linear expansion to d_ff and a linear contraction back to d_model.

    Args:
        d_model (int): The input and output dimension of the model.
        d_ff (int): The dimension of the inner hidden layer.
    """

    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding2D(nn.Module):
    """
    Generates 2D sinusoidal positional encodings for image patches.

    This module creates separate positional encodings for the height and width
    dimensions and concatenates them to provide 2D spatial information.

    Args:
        d_model (int): The dimension of the model. Must be divisible by 4.
        height (int): The number of patches along the height of the image.
        width (int): The number of patches along the width of the image.
    """
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        assert d_model % 2 == 0, "d_model must be an even number for 2D encoding."

        pe = torch.zeros(height * width, d_model)
        
        y_position = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        x_position = torch.arange(0, width, dtype=torch.float).unsqueeze(1)

        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2).float() * -(math.log(10000.0) / d_model_half))

        pe_y = torch.zeros(height, d_model_half)
        pe_y[:, 0::2] = torch.sin(y_position * div_term)
        pe_y[:, 1::2] = torch.cos(y_position * div_term)
        
        pe_x = torch.zeros(width, d_model_half)
        pe_x[:, 0::2] = torch.sin(x_position * div_term)
        pe_x[:, 1::2] = torch.cos(x_position * div_term)

        pe_y_expanded = pe_y.unsqueeze(1).repeat(1, width, 1)
        pe_x_expanded = pe_x.unsqueeze(0).repeat(height, 1, 1)

        pe = torch.cat([pe_y_expanded, pe_x_expanded], dim=2)
        
        self.register_buffer('pe', pe.view(-1, d_model).unsqueeze(0))

    def forward(self, x):

        return x + self.pe


class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder.

    Comprises a multi-head self-attention sub-layer and a position-wise
    feed-forward sub-layer. Residual connections and layer normalisation are
    applied around each sub-layer.

    Args:
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        d_ff (int): The dimension of the inner feed-forward layer.
        dropout (float): The dropout rate.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    """
    An encoder-only Transformer model for image-to-image tasks.

    This model takes an image, processes it as a sequence of patches through
    several encoder layers, and reconstructs an output image.

    Args:
        img_size (int): Size of the input image.
        patch_size (int): Size of each image patch.
        in_channels (int): Number of input image channels.
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of encoder layers to stack.
        d_ff (int): The dimension of the inner feed-forward layer.
        dropout (float): The dropout rate.
    """
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.d_model = config["d_model"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
 
        self.d_ff = config.get("d_ff", self.d_model * 4)
        self.dropout = config.get("dropout", 0.1)

        self.num_patches_per_dim = self.img_size // self.patch_size
        
        self.encoder_embedding = ImagePatchEmbedding(self.img_size, self.patch_size, self.in_channels, self.d_model)
        self.positional_encoding = PositionalEncoding2D(self.d_model, self.num_patches_per_dim, self.num_patches_per_dim)

        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)])

        self.dropout = nn.Dropout(self.dropout)

        patch_dim = self.in_channels * (self.patch_size ** 2)
        self.output_proj = nn.Linear(self.d_model, patch_dim)    

    def forward(self, src_img):

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src_img)))
        
        enc_output = src_embedded

        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask=None)

        output_patches = self.output_proj(enc_output)
        
        b, n, _ = output_patches.shape
        p = self.patch_size
        c = self.in_channels
        h = w = self.num_patches_per_dim

        output_patches = output_patches.view(b, h, w, c, p, p)
        output_patches = output_patches.permute(0, 3, 1, 4, 2, 5) # (b, c, h, p, w, p)
        output_image = output_patches.reshape(b, c, h * p, w * p)
        
        return output_image
    

class LearnablePosTransformer(nn.Module):
    """
    An image-to-image Transformer that uses learnable positional embeddings
    instead of fixed sinusoidal ones.

    Args:
        img_size (int): Size of the input image.
        patch_size (int): Size of each image patch.
        in_channels (int): Number of input image channels.
        d_model (int): The dimension of the model.
        num_heads (int): The number of attention heads.
        num_layers (int): The number of encoder layers to stack.
        d_ff (int): The dimension of the inner feed-forward layer.
        dropout (float): The dropout rate.
    """
    def __init__(self, config):
        super(LearnablePosTransformer, self).__init__()

        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.d_model = config["d_model"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]

        self.d_ff = config.get("d_ff", self.d_model * 4)
        self.dropout = config.get("dropout", 0.1)

        num_patches = (self.img_size // self.patch_size) ** 2

        self.positional_embedding = nn.Embedding(num_patches, self.d_model)
        self.encoder_embedding = ImagePatchEmbedding(self.img_size, self.patch_size, self.in_channels, self.d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout) for _ in range(self.num_layers)])
        
        self.dropout = nn.Dropout(self.dropout)
        
        patch_dim = self.in_channels * (self.patch_size ** 2)
        self.output_proj = nn.Linear(self.d_model, patch_dim)

        self.register_buffer('positions', torch.arange(num_patches).reshape(1, -1))
        
    def forward(self, src_img):

        patch_embeddings = self.encoder_embedding(src_img)
        
        pos_embedded = self.positional_embedding(self.positions)
        src_embedded = self.dropout(patch_embeddings + pos_embedded)
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask=None)

        output_patches = self.output_proj(enc_output)
        
        b, n, _ = output_patches.shape
        p = self.patch_size
        c = self.in_channels
        h = w = self.img_size // self.patch_size

        output_patches = output_patches.view(b, h, w, c, p, p)
        output_patches = output_patches.permute(0, 3, 1, 4, 2, 5) # (b, c, h, p, w, p)
        output_image = output_patches.reshape(b, c, h * p, w * p)
        
        return output_image
    

    
if __name__ == "__main__":
    import json
    with open("src/models/configs/transformer1.json", "r") as f:
        config = json.load(f)[0]
        transformer = Transformer(config)
        print(transformer)

'''#-------------------
# Example transformer usage

# parameters
img_size = config["img_size"]
in_channels = config["in_channels"]

transformer = Transformer(config)

batch_size = 64

src_image_data = torch.rand(batch_size, in_channels, img_size, img_size)

target_image_data = torch.rand(batch_size, in_channels, img_size, img_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(1):
    optimizer.zero_grad()

    output_image = transformer(src_image_data)

    loss = criterion(output_image, target_image_data)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

transformer.eval()

val_src_data = torch.rand(batch_size, in_channels, img_size, img_size)
val_target_data = torch.rand(batch_size, in_channels, img_size, img_size)

with torch.no_grad():
    val_output = transformer(val_src_data)
    val_loss = criterion(val_output, val_target_data)
    print(f"Validation Loss: {val_loss.item()}")'''