import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lightning as L
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig



class TextMaskerModel(L.LightningModule):
    def __init__(
        self, 
        hidden_size, 
        num_heads = 4, 
        num_layers = 2, 
        ff_size = 2048, 
        activation = "gelu", 
        dropout = 0.1,
        max_seq_length = 4096,
        skip_relu = False,
        skip_sigmoid = False,
        skip_rescale = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_size = ff_size
        self.activation = activation
        self.dropout = dropout
        self.max_seq_length = max_seq_length

        self.transformers = BertEncoder(BertConfig(
            hidden_size = hidden_size,
            num_hidden_layers = num_layers,
            num_attention_heads = num_heads,
            intermediate_size = ff_size,
            hidden_act = activation,
            hidden_dropout_prob = dropout,
            attention_probs_dropout_prob = dropout,
            _attn_implementation = "eager"
        ))
        self.linear = torch.nn.Linear(hidden_size, 1)
        self.linear_norm = torch.nn.LayerNorm(max_seq_length)
        
        # For ablation
        self.skip_relu = skip_relu
        self.skip_sigmoid = skip_sigmoid
        self.skip_rescale = skip_rescale

    def forward(self, embeds, attn_masks, conditioning=None):
        # Prepare mask for transformer
        attn_masks = attn_masks.unsqueeze(1).unsqueeze(3)
        attn_masks = torch.repeat_interleave(attn_masks, repeats=self.num_heads, dim=1)

        if conditioning is not None:
            embeds = embeds + conditioning.unsqueeze(1)

        # Compute token-wise embeddings
        embeds_masker = self.transformers(
            embeds, 
            attn_masks
        ).last_hidden_state

        batch_size, seq_length, _ = embeds_masker.shape
        
        out_weights = torch.zeros(batch_size, seq_length).to(embeds.device)
        sequence_lengths = attn_masks[:, 0, :, 0].sum(dim=1)

        for b in range(batch_size):
            sl = sequence_lengths[b]
            # Compute attribution scores
            raw_scores = self.linear(embeds_masker[b, :])[:, 0]
            raw_scores = F.relu(raw_scores) if not self.skip_relu else raw_scores
            raw_scores = self.linear_norm(raw_scores)
            raw_scores = F.sigmoid(raw_scores) if not self.skip_sigmoid else raw_scores
            out_weights[b, :sl] = (raw_scores[:sl] - raw_scores[:sl].min()) / (raw_scores[:sl].max() - raw_scores[:sl].min() + 1e-16) if not self.skip_rescale else raw_scores[:sl]
            
        return out_weights.unsqueeze(2)



class ImageMaskerModel(L.LightningModule):
    def __init__(
        self, 
        hidden_size,
        in_resolution,
        out_resolution,
        skip_relu = False,
        skip_sigmoid = False,
        skip_rescale = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_resolution = in_resolution
        self.out_resolution = out_resolution
    
        self.conv1_1 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.conv1_2 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(self.hidden_size),
        )
        
        self.conv2_1 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.conv2_2 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(self.hidden_size),
        )

        self.conv3_1 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.conv3_2 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(self.hidden_size),
        )

        self.conv4_1 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.conv4_2 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.BatchNorm2d(self.hidden_size),
        )
        
        self.conv_h1 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.conv_h2 = nn.Sequential( nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm2d(self.hidden_size), )
        self.head = nn.Conv2d(self.hidden_size, 1, kernel_size=1, stride=1)
        self.head_norm = nn.BatchNorm2d(1)

        # For ablation
        self.skip_relu = skip_relu
        self.skip_sigmoid = skip_sigmoid
        self.skip_rescale = skip_rescale

    def forward(self, hidden_states, conditioning=None, eps=1e-16):
        batch_size, _, _ = hidden_states[-1].shape
        activations = [ 
            hidden_states[i][:, 1:, :].permute(0, 2, 1).contiguous().reshape(batch_size, self.hidden_size, *self.in_resolution).float()
            for i in range(len(hidden_states))
        ]
        
        # Inject multimodal conditioning
        if conditioning is not None:
            conditioning = conditioning.unsqueeze(2).unsqueeze(3)
            activations = [ activ + conditioning for activ in activations ]


        out_weights = self.conv1_1(activations[-1])
        out_weights = self.conv1_2(out_weights)
        out_weights = self.upsample1(out_weights)

        skip_activ = torchvision.transforms.functional.resize(activations[-2], (self.in_resolution[0]*2, self.in_resolution[1]*2))
        out_weights = self.conv2_1(out_weights + skip_activ)
        out_weights = self.conv2_2(out_weights)
        out_weights = self.upsample2(out_weights)

        skip_activ = torchvision.transforms.functional.resize(activations[-3], (self.in_resolution[0]*4, self.in_resolution[1]*4))
        out_weights = self.conv3_1(out_weights + skip_activ)
        out_weights = self.conv3_2(out_weights)
        out_weights = self.upsample3(out_weights)

        skip_activ = torchvision.transforms.functional.resize(activations[-4], (self.in_resolution[0]*8, self.in_resolution[1]*8))
        out_weights = self.conv4_1(out_weights + skip_activ)
        out_weights = self.conv4_2(out_weights)
        out_weights = self.upsample4(out_weights)

        out_weights = self.conv_h1(out_weights)
        out_weights = self.conv_h2(out_weights)
        out_weights = self.head(out_weights)

        out_weights = F.relu(out_weights) if not self.skip_relu else out_weights
        out_weights = self.head_norm(out_weights)

        out_activ_shape = (out_weights.shape[2], out_weights.shape[3])
        out_masks = torch.zeros(batch_size, 1, *self.out_resolution).to(self.device)

        for b in range(batch_size):
            raw_scores = out_weights[b].flatten().reshape(1, 1, *out_activ_shape)
            raw_scores = F.sigmoid(raw_scores) if not self.skip_sigmoid else raw_scores
            raw_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-16) if not self.skip_rescale else raw_scores

            # Scale to out resolution
            raw_scores = torchvision.transforms.functional.resize(raw_scores, self.out_resolution, interpolation=torchvision.transforms.InterpolationMode.BILINEAR)
            raw_scores = raw_scores.flatten()

            # Put back to output resolution
            out_masks[b] = raw_scores.reshape(1, *self.out_resolution)
        
        return out_masks



class RandomTextMaskerModel(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, embeds, attn_masks):
        batch_size, seq_length, _ = embeds.shape
        out_weights = torch.rand(batch_size, seq_length).to(embeds.device)
        return out_weights.unsqueeze(2)



class RandomImageMaskerModel(L.LightningModule):
    def __init__(self, out_resolution):
        super().__init__()
        self.out_resolution = out_resolution

    def forward(self, hidden_states, eps=1e-16):
        batch_size, _, _ = hidden_states[-1].shape
        out_masks = torch.rand(batch_size, 1, *self.out_resolution).to(self.device)
        return out_masks