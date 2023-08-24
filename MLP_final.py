#Imports
import torch
import torch.nn as nn

#Layers building block
class MLP(nn.Module):
    def __init__(self, in_feature, out_feature, dropout):
        super().__init__()
        self.in_feature = in_feature
        self.dropout = dropout
        self.Linear=nn.Linear(in_feature,out_feature)
        self.dropout = nn.Dropout(p=self.dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        skip_x = x
        x = self.Linear(x)
        x = self.dropout(x)
        x = x+skip_x
        x = self.activation(x)

        return x

#Fingerprint MLP model
class FpMLP(nn.Module):
    def __init__(self, args):
        super(FpMLP, self).__init__()

        # Argument Define
        self.dim_of_fp = args["fp_dim"]
        self.dim_of_Linear = args["hidden_dim"]

        self.N_predict_layer = args["N_MLP_layer"]
        self.N_predict_FC = args["N_predictor_layer"]

        self.N_properties = args["N_properties"]

        self.dropout = args["dropout"]

        self.embedding=nn.Linear(self.dim_of_fp,self.dim_of_Linear)

        self.MLPs= nn.ModuleList([
            MLP(self.dim_of_Linear,self.dim_of_Linear,self.dropout) for _ in range(self.N_predict_layer)])

        self.predict = \
            nn.ModuleList([
                nn.Sequential(nn.Linear(self.dim_of_Linear,self.dim_of_Linear),
                              nn.Dropout(p=self.dropout),
                              nn.ReLU())
                for _ in range(self.N_predict_FC-1)] +
                [nn.Linear(self.dim_of_Linear,self.N_properties)
            ])

    def forward(self, x):
        x = self.embedding(x)

        for layer in self.MLPs:
            x = layer(x)

        for layer in self.predict:
            x = layer(x)
        return x