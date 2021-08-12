"""
download pretrained bert-base-nli-stsb-mean-tokens BERT to pretrained_bert and save it in SBERT format
"""
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util
import torch
from torch import nn, Tensor, device

class SBERT(nn.Module):
    """entity-aware BERT
        also supports regular SBERT without entity

        esbert_model can be either entity_transformer or regular transformer
    """
    
    def __init__(self, bert_model, device="cuda"):
        super(SBERT, self).__init__()
        self.bert_model = bert_model.to(device)
            
    def forward(self, features):
        
        batch_to_device(features, device)
        bert_features = self.bert_model(features)
        cls_embeddings = bert_features['cls_token_embeddings']
        token_embeddings = bert_features['token_embeddings']
        attention_mask = bert_features['attention_mask']
        
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1) # tokens not weighted
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        # print(output_vector.shape)

        features.update({"sentence_embedding": output_vector})
        return features

class EntitySBert(nn.Module):
    """entity-aware BERT"""
    
    def __init__(self, esbert_model, device="cuda"):
        super(EntitySBert, self).__init__()
        self.esbert_model = esbert_model.to(device)
            
    def forward(self, features):
        
        batch_to_device(features, device)
        bert_features = self.esbert_model(features)
        cls_embeddings = bert_features['cls_token_embeddings']
        token_embeddings = bert_features['token_embeddings']
        attention_mask = bert_features['attention_mask']
        
        output_vectors = []
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1) # tokens not weighted
        output_vectors.append(sum_embeddings / sum_mask)
        output_vector = torch.cat(output_vectors, 1)
        # print(output_vector.shape)

        features.update({"sentence_embedding": output_vector})
        return features


# save the pre-trained model
model_name = 'bert-base-nli-stsb-mean-tokens'
sbert_hugginface = SentenceTransformer(model_name)
sbert_hugginface.save("pretrained_bert")

# also save the model in our format
torch.save(sbert_hugginface, "./pretrained_bert/bert-base-nli-stsb-mean-tokens.pt")

sbert_hugginface = torch.load("./pretrained_bert/bert-base-nli-stsb-mean-tokens.pt")
sbert_ours = SBERT(sbert_hugginface[0]) # sbert, without entity
torch.save(sbert_ours, "./pretrained_bert/SBERT-base-nli-stsb-mean-tokens.pt")
