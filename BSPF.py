import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class bspf_network_vanilla(nn.Module):
    def __init__(self, embedding_size, num_planes):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_planes = num_planes


        # the below is going to be inertible
        self.plane_feat_size = 4
        self.output_size = self.num_planes * 4
        self.mlp = nn.Linear(embedding_size * 2, self.output_size)

        # initializing both to be 0
        nn.init.constant_(self.mlp.weight, 0)
        nn.init.constant_(self.mlp.bias, 0)

    def forward(self, tgt_z, src_z, src_planes):
        plane_delta = self.mlp(torch.cat([tgt_z, src_z], dim=1)) # batchsize x 4 
        plane_delta = plane_delta.reshape(plane_delta.shape[0], -1, self.num_planes)
        return src_planes + plane_delta
        # needs to be output 
        # batchsize x num_planes x 4

class bspf_network(nn.Module):
    """
    This implementation uses a transformer, and treats the planes like a sequence of tokens.
    This should in theory be equivalent to a dense graph neural network.
    """
    def __init__(self, embedding_size, num_planes):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.num_planes = num_planes

        self.z_size = 128  # TODO load from hyperparameters
        self.hidden_dims = 128
        self.nheads = 1
        self.dropout = 0.0
        self.nlayers = 1
        self.primitive_params = 4 

        self.feature_extractor = nn.Linear(embedding_size*2, self.z_size) # [batch  x (2*embedding_size)]  --> [batch x z_size]
        self.transformer = MyTransformerEncoder(self.z_size + self.primitive_params, 
                                                self.hidden_dims, self.nheads, 
                                                self.dropout, self.nlayers)
        self.flow = nn.Linear(self.z_size + self.primitive_params, self.primitive_params) # [batch x z_size] --> [batch x num_planes x 4] 


        nn.init.constant_(self.flow.weight, 0)
        nn.init.constant_(self.flow.bias, 0)



    def extract_features(self, tgt_z, src_z, src_planes):
        """
        for extracting the feature uesd to create the flow field.
        src_planes: (batch_size,  num_planes, parameters)
        """
        feat = self.feature_extractor(torch.cat([tgt_z,src_z], dim=1)) # batchsize * z_size
        # put into transformer now
        feat = torch.cat([feat.unsqueeze(1).repeat(1, self.num_planes, 1),
                          src_planes], dim=-1) # batchsize x numplanes x [feat_size]
        
        # the sequence goes : numplanes x batchsize x [feat_size]
        out = self.transformer(feat.transpose(0,1)).transpose(0,1)
        return out

    def forward(self, tgt_z, src_z, src_planes):
        """
        src_planes: (batch_size, parameters, num_planes)
        """
        src_planes = src_planes.transpose(1,2) 
        feat = self.extract_features(tgt_z, src_z, src_planes)
        # feat = torch.cat([feat.reshape(-1, feat.shape[-1]), src_planes.transpose(1,2).reshape(-1, self.primitive_params)], dim=-1)
        out = self.flow(feat.reshape(-1, feat.shape[-1])).reshape(feat.shape[0], feat.shape[1], -1) 
        # tile so that it's (batch * self.num_planes,  zsize) (duplicate across the number of planes)
        out = out.transpose(1,2) # -> batchsize, parameters, num_planes

        # now regress.  NOTE: single step for now.
        out = src_planes.transpose(1,2) + out
        return out

class PositionalEncoding(nn.Module):                                                                     
                                                                                                         
    def __init__(self, d_model, dropout=0.1, max_len=5000):                                              
        super(PositionalEncoding, self).__init__()                                                       
        self.dropout = nn.Dropout(p=dropout)                                                             
                                                                                                         
        pe = torch.zeros(max_len, d_model)                                                               
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)                              
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))       
        pe[:, 0::2] = torch.sin(position * div_term)                                                     
        pe[:, 1::2] = torch.cos(position * div_term)                                                     
        pe = pe.unsqueeze(0).transpose(0, 1)                                                             
        self.register_buffer('pe', pe)                                                                   
                                                                                                         
    def forward(self, x):                                                                                
        x = x + self.pe[:x.size(0), :]                                                                   
        return self.dropout(x)                                                                           


class MyTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, nheads, dropout, nlayers):                                                                
        super().__init__()                                                                               
        
        self.embedding_dim = embedding_dim                                                               
        self.hidden_dim = hidden_dim                                                                     
        self.nhead = nheads                                                                              
        self.dropout = dropout                                                                           
        self.nlayers = nlayers                                                                           
                                                                                                         
        # self.token_embeddings = nn.Linear(self.vocab_size, self.embedding_dim, bias=False)               
        self.pos_encoder = PositionalEncoding(self.embedding_dim,                                        
                                              self.dropout)                                              
        self.encoder_layers = nn.TransformerEncoderLayer(self.embedding_dim,                             
                                                         self.nhead,                                     
                                                         dim_feedforward=self.hidden_dim,                
                                                         dropout=self.dropout)                           
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers,                            
                                                         self.nlayers)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, seq):
        """
        seq is a onehot encoding of the tokens. [seq x batch x word_dim]
        """
        # program_emb = self.token_embeddings(seq)
        # program_emb *= math.sqrt(self.embedding_dim) 
        program_emb = self.pos_encoder(seq)
        output = self.transformer_encoder(program_emb)
        return output



                                                                                                         