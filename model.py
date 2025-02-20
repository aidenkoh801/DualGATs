import math
from model_utils import RGAT, DiffLoss
import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleGATs(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        
        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        # Add Transformer Module
        self.TransformerLayers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.hidden_dim,
                nhead=args.transformer_heads,
                dim_feedforward=args.hidden_dim*4,
                dropout=args.dropout
            ) for _ in range(args.gnn_layers)
        ])
        
        # Add positional encoding
        self.pos_encoder = PositionalEncoding(
            args.hidden_dim,
            args.dropout,
            max_len=1000  # Adjust based on max utterance length
        )


        SpkGAT = []
        DisGAT = []
        for _ in range(args.gnn_layers):
            SpkGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=6))
            DisGAT.append(RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=18))

        self.SpkGAT = nn.ModuleList(SpkGAT)
        self.DisGAT = nn.ModuleList(DisGAT)


        self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)
        # Add new affine parameters for Transformer cross-attention
        self.affine3 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine3.data, gain=1.414)
        self.affine4 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine4.data, gain=1.414)


        self.diff_loss = DiffLoss(args)
        self.beta = 0.3

        #in_dim = args.hidden_dim *2 + args.emb_dim
        # Update input dimension: 3*hidden_dim + emb_dim
        in_dim = args.hidden_dim *3 + args.emb_dim  
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.drop = nn.Dropout(args.dropout)

       

    # def forward(self, utterance_features, semantic_adj, structure_adj):
    #     '''
    #     :param tutterance_features: (B, N, emb_dim)
    #     :param xx_adj: (B, N, N)
    #     :return:
    #     '''
    #     batch_size = utterance_features.size(0)
    #     H0 = F.relu(self.fc1(utterance_features)) # (B, N, hidden_dim)
    #     H = [H0]
    #     diff_loss = 0
    #     for l in range(self.args.gnn_layers):
    #         if l==0:
    #             H1_semantic = self.SpkGAT[l](H[l], semantic_adj)
    #             H1_structure = self.DisGAT[l](H[l], structure_adj)
    #         else:
    #             H1_semantic = self.SpkGAT[l](H[2*l-1], semantic_adj)
    #             H1_structure = self.DisGAT[l](H[2*l], structure_adj)


    #         diff_loss = diff_loss + self.diff_loss(H1_semantic, H1_structure)
    #         # BiAffine 

    #         A1 = F.softmax(torch.bmm(torch.matmul(H1_semantic, self.affine1), torch.transpose(H1_structure, 1, 2)), dim=-1)
    #         A2 = F.softmax(torch.bmm(torch.matmul(H1_structure, self.affine2), torch.transpose(H1_semantic, 1, 2)), dim=-1)

    #         H1_semantic_new = torch.bmm(A1, H1_structure)
    #         H1_structure_new = torch.bmm(A2, H1_semantic)

    #         H1_semantic_out = self.drop(H1_semantic_new) if l < self.args.gnn_layers - 1 else H1_semantic_new
    #         H1_structure_out = self.drop(H1_structure_new) if l <self.args.gnn_layers - 1 else H1_structure_new


    #         H.append(H1_semantic_out)
    #         H.append(H1_structure_out)

    #     H.append(utterance_features) 

    #     H = torch.cat([H[-3],H[-2],H[-1]], dim = 2) #(B, N, 2*hidden_dim+emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行
    #     logits = self.out_mlp(H)
    #     return logits, self.beta * (diff_loss/self.args.gnn_layers)

    def forward(self, utterance_features, semantic_adj, structure_adj):
        batch_size, seq_len = utterance_features.size(0), utterance_features.size(1)
        
        
        H0 = F.relu(self.fc1(utterance_features)) # linear layer to reduce RoBERTa's emb_dim to hidden_dim
        H0 = self.pos_encoder(H0) # Add positional encoding
        H = [H0]
        diff_loss = 0

        for l in range(self.args.gnn_layers):
            if l == 0:
                H_s = self.SpkGAT[l](H[-1], semantic_adj)
                H_d = self.DisGAT[l](H[-1], structure_adj)
                H_t = self.TransformerLayers[l](H[-1].permute(1,0,2)).permute(1,0,2)
            else:
                H_s = self.SpkGAT[l](H_s_n, semantic_adj)
                H_d = self.DisGAT[l](H_d_n, structure_adj)
                H_t = self.TransformerLayers[l](H_t_n.permute(1,0,2)).permute(1,0,2)

            # Update diff loss for three pairs
            diff_loss += (
                self.diff_loss(H_s, H_d) + 
                self.diff_loss(H_s, H_t) + 
                self.diff_loss(H_d, H_t)
            )

            # Cross attention between all three components
            # 1. Spk <-> Dis
            A_sd = F.softmax(torch.bmm(torch.matmul(H_s, self.affine1), 
                                     H_d.transpose(1,2)), dim=-1)
            A_ds = F.softmax(torch.bmm(torch.matmul(H_d, self.affine2),
                                     H_s.transpose(1,2)), dim=-1)
            
            # 2. Add Transformer cross-attention
            A_st = F.softmax(torch.bmm(torch.matmul(H_s, self.affine3),
                                     H_t.transpose(1,2)), dim=-1)
            A_ts = F.softmax(torch.bmm(torch.matmul(H_t, self.affine4),
                                     H_s.transpose(1,2)), dim=-1)
            
            # 3. Transform cross-attention
            A_dt = F.softmax(torch.bmm(torch.matmul(H_d, self.affine3),
                                     H_t.transpose(1,2)), dim=-1)
            A_td = F.softmax(torch.bmm(torch.matmul(H_t, self.affine4),
                                     H_d.transpose(1,2)), dim=-1)
            
            # Update representations
            H_s_n = (torch.bmm(A_sd, H_d) + torch.bmm(A_st, H_t)) / 2
            H_d_n = (torch.bmm(A_ds, H_s) + torch.bmm(A_dt, H_t)) / 2
            H_t_n = (torch.bmm(A_ts, H_s) + torch.bmm(A_td, H_d)) / 2

            # Store outputs
            H.append(self.drop(H_s_n))
            H.append(self.drop(H_d_n)) 
            H.append(self.drop(H_t_n))

        # Final concatenation: last layer outputs + original features
        H = torch.cat([
            H[-3],  # Last spk
            H[-2],  # Last dis
            H[-1],  # Last transformer
            utterance_features
        ], dim=2)
        
        logits = self.out_mlp(H)
        return logits, self.beta * (diff_loss/self.args.gnn_layers)

# Add positional encoding module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



