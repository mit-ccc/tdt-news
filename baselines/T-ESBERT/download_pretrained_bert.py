from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util

model_name = 'bert-base-nli-stsb-mean-tokens'
model = SentenceTransformer(model_name)
model.save("pretrained_bert")