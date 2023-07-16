import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def vector_angles(positions, dimensions, encoding_size):
    return positions / 10000**((2*dimensions) / encoding_size)


def positional_encoding(max_positions, encoding_size):
    pos_indices = np.expand_dims(np.arange(max_positions), axis = 1)
    encoding_indices = np.expand_dims(np.arange(encoding_size), axis = 0)
    angles = vector_angles(pos_indices, encoding_indices, encoding_size)
    angles[:,0::2] = np.sin(angles[:, 0::2])
    angles[:,1::2] = np.cos(angles[:, 1::2])
    return torch.from_numpy(np.expand_dims(angles, axis = 0))


def attention_mask(seq_length):
    return torch.tril(torch.ones((1, seq_length, seq_length)))


def fc_layer(embed_dim, fc_dim):
    return nn.Sequential(
        nn.Linear(fc_dim, fc_dim),
        nn.ReLU(),
        nn.Linear(fc_dim, embed_dim),
        nn.ReLU(),
    )


class Encoder_Layer(nn.Module):
    def __init__(self, embedding_size, num_heads, fc_dim, dropout_ = 0.2, epsilon = 1e-6):
        super(Encoder_Layer, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim = embedding_size,
num_heads = num_heads, 
                                                          dropout = dropout_, 
                                                          kdim = embedding_size)
        self.fcn = fc_layer(embed_dim = embedding_size, fc_dim = fc_dim)
        self.layernorm1 = nn.LayerNorm(normalized_shape = embedding_size, eps = epsilon)
        self.layernorm2 = nn.LayerNorm(normalized_shape = embedding_size, eps = epsilon)
        self.dropout = nn.Dropout(p = dropout_)
    
    def forward(self, x, mask, training = 'train'):
        multiheadatt_output, selfatt_output = self.multi_head_attention(x, x, x, attn_mask = mask)
        multiheadatt_output = self.layernorm1(x + multiheadatt_output)
        fc_output = self.fcn(multiheadatt_output)
        if training == 'train':
            fc_output = self.dropout(fc_output)
            
        fc_output = self.layernorm2(multiheadatt_output + fc_output)
        return fc_output, selfatt_output


class Encoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, fc_dim, input_vocab_size, max_pos
                ,dropout_= 0.2, epsilon = 1e-6):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(num_embeddings = input_vocab_size, embedding_dim = self.embed_dim)
        self.pos_encoding = positional_encoding(max_pos, self.embed_dim)
        self.encode_layers = nn.ModuleList([Encoder_Layer(embedding_size = embed_dim, 
                                             num_heads = num_heads,
                                             fc_dim = fc_dim,
                                             dropout_ = dropout_,
                                             epsilon = epsilon) 
                              for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(p = dropout_)
        
    def forward(self, x, mask, training = 'train'):
        self_attention_values = {}
        seq_length = x.shape[1]
        x = self.embedding(x)
        x *= torch.sqrt(self.embed_dim)
        x += self.pos_encoding[:, :seq_length, :]
        if training == 'train':
            x = self.dropout(x)

        for i in range(self.num_layers):
            x, selfattn_value = self.encode_layers[i](x, mask, training = 'train')
            self_attention_values[f'Encoder Layer {i+1} Attention Values'] = selfattn_value   
        return x, self_attention_values
    
class Decoder_Layer(nn.Module):
    def __init__(self, embedding_size, num_heads, fc_dim, dropout_ = 0.2, epsilon = 1e-6):
        super(Decoder_Layer, self).__init__()
        self.multi_head_attention1 = nn.MultiheadAttention(embed_dim = embedding_size,
num_heads = num_heads, 
                                                          dropout = dropout_, 
                                                          kdim = embedding_size)
        self.multi_head_attention2 = nn.MultiheadAttention(embed_dim = embedding_size,
num_heads = num_heads, 
                                                          dropout = dropout_, 
                                                          kdim = embedding_size)
        self.fcn = fc_layer(embed_dim = embedding_size, fc_dim = fc_dim)
        self.layernorm1 = nn.LayerNorm(normalized_shape = embedding_size, eps = epsilon)
        self.layernorm2 = nn.LayerNorm(normalized_shape = embedding_size, eps = epsilon)
        self.layernorm3 = nn.LayerNorm(normalized_shape = embedding_size, eps = epsilon)
        self.dropout = nn.Dropout(p = dropout_)
    
    def forward(self, x, mask, encoder_output, training = 'train'):
        multiheadatt1_output, selfatt_output = self.multi_head_attention1(x, x, x, attn_mask = mask)
        query = self.layernorm1(x +  multiheadatt1_output)
        multiheadatt2_output, dec_enc_att_output = self.multi_head_attention2(query, encoder_output, 
                                                                           encoder_output, 
                                                                           attn_mask = mask)
        multiheadatt2_output = self.layernorm2(query + multiheadatt2_output)
        fc_output = self.fcn(multiheadatt2_output)
        if training == 'train':
            fc_output = self.dropout(fc_output)
        
        fc_output = self.layernorm3(multiheadatt2_output + fc_output)
        return fc_output, selfatt_output, dec_enc_att_output
    
class Decoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, fc_dim, target_vocab_size, max_pos
                ,dropout_= 0.2, epsilon = 1e-6):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings = target_vocab_size, embedding_dim = self.embed_dim)
        self.pos_encoding = positional_encoding(max_pos, self.embed_dim)
        self.decode_layers = nn.ModuleList([Decoder_Layer(embedding_size = embed_dim, 
                                             num_heads = num_heads,
                                             fc_dim = fc_dim,
                                             dropout_ = dropout_,
                                             epsilon = epsilon) 
                              for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(p = dropout_)
        
    def forward(self, x, mask, encoder_output, training='train'):
        attention_values = {}
        seq_length = x.shape[1]
        x = self.embedding(x)
        x *= torch.sqrt(self.embed_dim)
        x += self.pos_encoding[:, :seq_length, :]
        if training == 'train':
            x = self.dropout(x)
        
        for i in range(self.num_layers):
            x, selfatt_values, dec_enc_att_values = self.decode_layers[i](x, 
                                                                          mask, 
                                                                          encoder_output, 
                                                                          training='train')
            attention_values[f'Decoder Layer {i+1} Self Attention Values'] = selfatt_values
            attention_values[f'Decoder Layer {i+1} Dec-Enc Attention values'] = dec_enc_att_values
        return x, attention_values
    
class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, fc_dim, input_vocab_size, 
                 target_vocab_size, max_pos_input, max_pos_output, dropout_= 0.2, epsilon = 1e-6):
        super(Transformer, self).__init__()
        self.encoder_block = Encoder(num_layers = num_layers,
                               embed_dim = embed_dim,
                               fc_dim = fc_dim,
                               input_vocab_size = input_vocab_size,
                               max_pos = max_pos_input,
                               dropout_ = dropout_,
                               epsilon = epsilon)
        
        self.decoder_block = Decoder(num_layers = num_layers,
                               embed_dim = embed_dim,
                               fc_dim = fc_dim,
                               target_vocab_size = target_vocab_size,
                               max_pos = max_pos_output,
                               dropout_ = dropout_,
                               epsilon = epsilon)
        
        self.output_layer = nn.Sequential(
                             nn.Linear(fc_dim, target_vocab_size),
                             nn.LogSoftmax(dim = -1)
                            )
    def forward(self, mask, input_seq, output_seq, training = 'train'):
        encoder_output, encoder_self_attention_values = self.encoder_block(input_seq, mask = None, 
                                                                           training = 'train')
        decoder_output, decoder_attention_values = self.decoder_block(output_seq,
                                                                      mask,
                                                                      encoder_output, 
                                                                      training = 'train')
        output = self.output_layer(decoder_output)
        return output, encoder_self_attention_values, decoder_attention_values
