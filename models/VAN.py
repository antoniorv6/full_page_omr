from turtle import forward
from numpy import dtype
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchinfo import summary
from torch.nn.init import zeros_, ones_, kaiming_uniform_

def VAN_Weight_Init(m):
    """
    Weights initialization for model training from scratch
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        if m.weight is not None:
            kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            ones_(m.weight)
        if m.bias is not None:
            zeros_(m.bias)

class DepthSepConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1,1), dilation=(1,1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None
        
        if padding:
            if padding is True:
                padding = [int((k-1)/2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)

        self.depth_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1,1))
        self.activation = activation

    def forward(self, inputs):
        x = self.depth_conv(inputs)
        if self.padding:
            x = F.pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        
        x = self.point_conv(x)

        return x

class MixDropout(nn.Module):
    def __init__(self, dropout_prob=0.4, dropout_2d_prob=0.2):
        super(MixDropout, self).__init__()

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2D = nn.Dropout2d(dropout_2d_prob)
    
    def forward(self, inputs):
        if random.random() < 0.5:
            return self.dropout(inputs)
        return self.dropout2D(inputs)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=(1,1), kernel=3, activation=nn.ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=kernel, padding=kernel//2)
        self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=(3,3), padding=(1,1), stride=stride)
        self.normLayer = nn.InstanceNorm2d(num_features=out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, inputs):
        pos = random.randint(1,3)

        x = self.conv1(inputs)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)
        
        x = self.normLayer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x

class DSCBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=(2, 1), activation=nn.ReLU, dropout=0.5):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_c, out_c, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = nn.InstanceNorm2d(out_c, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_prob=dropout, dropout_2d_prob=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        #x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class Encoder(nn.Module):

    def __init__(self, in_channels, dropout=0.4):
        super(Encoder, self).__init__()

        self.conv_blocks = nn.ModuleList([
            ConvBlock(in_c=in_channels, out_c=16, stride=(1,1), dropout=dropout),
            ConvBlock(in_c=16, out_c=32, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=32, out_c=64, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=64, out_c=128, stride=(2,2), dropout=dropout),
            ConvBlock(in_c=128, out_c=128, stride=(2,1), dropout=dropout),
            ConvBlock(in_c=128, out_c=128, stride=(2,1), dropout=dropout),
        ])

        self.dscblocks = nn.ModuleList([
            DSCBlock(in_c=128, out_c=128, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=128, out_c=128, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=128, out_c=128, stride=(1,1), dropout = dropout),
            DSCBlock(in_c=128, out_c=256, stride=(1,1), dropout = dropout)
        ])
    
    def forward(self, x):
        for layer in self.conv_blocks:
            x = layer(x)
        
        for layer in self.dscblocks:
            xt = layer(x)
            x = x + xt if x.size() == xt.size() else xt

        return x
    

class VerticalAttention(nn.Module):
    
    def __init__(self, min_width, encoder_features, att_fc_size, hidden_size, dropout=0):
        
        super(VerticalAttention, self).__init__()

        self.att_fc_size = att_fc_size

        self.ada_pool = nn.AdaptiveAvgPool2d((None, min_width))
        self.dense_width = nn.Linear(min_width, 1)
        
        self.dense_enc = nn.Linear(encoder_features, att_fc_size)
        self.dense_align = nn.Linear(att_fc_size, 1)

        self.norm = nn.InstanceNorm1d(2, track_running_stats=False)
        self.conv_block = nn.Conv1d(2, 16, kernel_size=15, padding=7)
        self.dense_conv_block = nn.Linear(16, att_fc_size)
        self.dense_hidden = nn.Linear(hidden_size, att_fc_size)
        self.dropout = nn.Dropout(dropout)

        self.h_features = None
    
    def forward(self, features, prev_attn_weights, coverage_vector=None, hidden=None, status='init'):
        
        # features (batch_size, features, height, width)
        # prev_attn_weights (batch_size, height)
        # coverage_vector (batch_size, height)
        # hidden (num_decoder_layers, batch_size, hidden_size)

        if status == "reset":
            self.h_features = self.h_features.detach()
        if status in ["init", "reset", ]:
            self.h_features = self.ada_pool(features) #Reduce la imagen a un ancho fijo (ellos establecen un valor de 100 en el paper) --> (b, f, h//32, 100)
            self.h_features = self.dense_width(self.h_features).squeeze(3) #Colapsamos el ancho (3.2.1 en el paper). Por lo tanto, sacamos un vector tipo (b, f, h//32)
        
        b, c, h, w = features.size()
        device = features.device

        
        # Creamos el vector de información local-global --> Fórmula 3 del paper
        
        sum = torch.zeros((b, h, self.att_fc_size), dtype=features.dtype, device=device) # Inicializamos la matriz de atención a 0
        cat = list()

        cat.append(prev_attn_weights) # Añadimos los pesos inmediatamente anteriores de la atención
        cat.append(torch.clamp(coverage_vector, 0, 1)) # Añaidmos la suma de todas las líneas ya procesadas (matrices de atención) y se limita entre 0 y 1. Se hace por estabilidad según el paper
        # cat = (2, bs, height)

        # Término 1 -> Wf * f'i = Features de la propia imagen (información global)
        enc_features_projected = self.dense_enc(self.h_features.permute(0,2,1)) # Proyectamos las features verticales al tamaño de nuestra matriz de atención -> (bs, h//32, att_f)
        sum += self.dropout(enc_features_projected) #Añadimos estas features verticales a la matriz de atención

        # Término 2 -> Wj * Jti = Información descrita por las matrices de atención producidas previamente (información local)
        cat = torch.cat([c.unsqueeze(1) for c in cat], dim=1) #(2, bs, height) -> (bs, 2, height)
        cat = self.norm(cat) 
        cat = self.conv_block(cat) # (bs, 2, height) -> (bs, 16, height)
        cat_projection = self.dense_conv_block(cat.permute(0,2,1)) # (bs, height, att_f)
        sum += self.dropout(cat_projection) 

        # Término 3 -> Wh * Hwf(t-1) = Último estado del decoder, donde se da información de todos los caracteres predichos anteriormete (información global)
        hidden_projection = self.dense_hidden(hidden[0]).permute(1,0,2) #Añadimos los estados internos de la lstm del decoder
        sum += self.dropout(hidden_projection)
        
        # Ahora mismo sum es el término Sij con toda la información contextual sobre el proceso de reconocimiento

        sum = torch.tanh(sum) # Aplicamos TanH para pronunciar los elementos más destacables
        align_score = self.dense_align(sum) # Producimos las puntuaciones de atención para el paso t
        attn_weights = F.softmax(align_score, dim=1) # Usamos una función softmax para dar una distribución probabilística a la matriz de atención
        #(bs, h//32, att_f)
        
        context_vector = torch.matmul(features.permute(0,1,3,2), attn_weights.unsqueeze(1)).squeeze(3) 
        # Multiplicamos la matriz de atención por las features obtenidas de la convlucional -> (bs, c, w//8, h//32) x (bs, 1, h//32, att_f) = (bs, att_f, w//8, c) = (bs, att_f, w//8)

        decision = None # Esto es por si la red aprende a detectar el final del párrafo, pero no es lo que mejor les va

        return context_vector, attn_weights.squeeze(2), decision


class Decoder(nn.Module):

    def __init__(self, encoder_size, hidden_size, out_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(encoder_size, hidden_size, num_layers=1)
        self.end_conv = nn.Conv2d(hidden_size, out_size, kernel_size=1)

    def forward(self, x, h=None):
        
        x, h = self.lstm(x.permute(2,0,1), h)
        x = x.permute(1,2,0)
        out = self.end_conv(x.unsqueeze(3)).squeeze(3)
        out = torch.squeeze(out, dim=2)
        out = F.log_softmax(out, dim=1)

        return out, h

class VAN(nn.Module):

    def __init__(self, input_channels, min_width, encoder_features, att_fc_size, attn_hidden, decoder_size, out_cats, device):
        super(VAN, self).__init__()
        self.device_creation = device
        self.encoder = Encoder(in_channels=input_channels)
        self.vertical_attention = VerticalAttention(min_width, encoder_features, att_fc_size, attn_hidden)
        self.decoder = Decoder(attn_hidden, decoder_size, out_cats)
        self.status = "init"
        self.attention_weights = None
        self.coverage = None
        self.hidden = None
    
    def forward(self, x):
        x = self.forward_encoder(x)
        x = self.forward_decoder_pass(x)
        return x

    def forward_encoder(self, x):
        return self.encoder(x)
    
    def forward_decoder_pass(self, x):
        if self.status == "init":
            b, c, h, w = x.size()
            self.attention_weights = torch.zeros((b, h), dtype=torch.float, device=self.device_creation)
            self.coverage = self.attention_weights.clone()
            self.hidden =torch.zeros((1, b, 256), device=self.device_creation), torch.zeros((1, b, 256), device=self.device_creation)

        self.context_vector, self.attention_weights, _ = self.vertical_attention(x, self.attention_weights, self.coverage, self.hidden, status=self.status)
        self.status = "inprogress"
        self.coverage = self.coverage + self.attention_weights # Formula 2
        prediction, self.hidden = self.decoder(self.context_vector, self.hidden)
        return prediction
    
    def get_attention_weights(self):
        return self.attention_weights
    
    def reset_status_forward(self):
        self.status = "init"

def get_VAN_model(min_width, encoder_features, attention_fc_size, attention_hidden, decoder_size, out_cats, input_channels=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    van_model = VAN(input_channels, min_width, encoder_features, attention_fc_size, attention_hidden, decoder_size, out_cats, device).to(device)
    van_model.apply(VAN_Weight_Init)
    summary(van_model, input_size=[(1,input_channels,1418,922)], dtypes=[torch.float])
    return van_model, device

if __name__ == "__main__":
    get_VAN_model(1024, 1024, 100, 256, 256, 256, 92, 512)
