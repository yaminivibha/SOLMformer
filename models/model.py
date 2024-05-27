import torch
import torch.nn as nn
from transformers import AutoModel
from models.gpt import GPTLanguageModel, LanguageDurationModel, EventMetadataLanguageModel

class MetadataSequenceModel(nn.Module):
    def __init__(
        self, 
        numerical_metadata_dim: int,
        hidden_dim: int,
        activity_vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
    ) -> None:
        
        self.numerical_metadata_dim = numerical_metadata_dim
        super().__init__()
        
        # Setting the metadata embedding dimension, need to add 1 if odd (so it adds up)
        self.text_metadata_embedder = TextMetaDataEmbeddingModel(n_embd//2)
        self.numerical_embedder = nn.Sequential(
            nn.Linear(numerical_metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd//2)
        )
        self.LanguageModel = GPTLanguageModel(activity_vocab_size, n_embd, n_head, n_layer, block_size)
    
    def forward(self, metadata_text: str, metadata_numbers: torch.Tensor, activity_sequence: torch.Tensor):
        text_embedding = self.text_metadata_embedder(metadata_text)
        numerical_embedding = self.numerical_embedder(metadata_numbers)
        metadata_embedding = torch.cat((text_embedding, numerical_embedding), -1)
        return self.LanguageModel(activity_sequence, metadata_embedding)
    
    def generate(self, activity_sequence: torch.Tensor, metadata_text: str=None, metadata_numbers:torch.Tensor=None, max_new_tokens: int=1, terminal_code:int=22, pad_code:int=23, max_likelihood:bool=False):
        if metadata_text is not None and metadata_numbers is not None:
            text_embedding = self.text_metadata_embedder(metadata_text)
            numerical_embedding = self.numerical_embedder(metadata_numbers.float())
            metadata_vector = torch.cat((text_embedding, numerical_embedding), -1)
            return self.LanguageModel.generate(activity_sequence, metadata_vector, max_new_tokens, terminal_code, pad_code, max_likelihood)
        else:
            return self.LanguageModel.generate(activity_sequence, None, max_new_tokens, terminal_code, pad_code, max_likelihood)

class TextMetaDataEmbeddingModel(nn.Module):
    def __init__(self, text_metadata_dim:int,  pretrained_model_name="huawei-noah/TinyBERT_General_4L_312D") -> None:
        
        self.text_metadata_dim = text_metadata_dim
        super().__init__()

        # Initialize the BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        # allow the BERT model to be finetuned
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.linear = nn.Linear(self.bert.config.hidden_size, text_metadata_dim)

    def forward(self, tokens):
        # Generate attention mask
        b, m, e = tokens['input_ids'].shape
        attention_mask = tokens["attention_mask"].squeeze(-1)
        input_ids = tokens['input_ids'].reshape(b * m, e)
        attention_mask = attention_mask.reshape(b * m, e)
        
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask) # B x (metadata length) x (embedding dimension)
        embeddings = outputs.last_hidden_state[:, 0, :] # B x (embedding dimension) # why using the first vector of last hidden state?
        # reshape to the text_metadata_dim
        output_embedding = self.linear(embeddings)
        return output_embedding

class DurationModel(nn.Module):
    def __init__(
        self, 
        numerical_metadata_dim: int,
        duration_embedding_dim: int,
        hidden_dim: int,
        activity_vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        pad_code: int,
        duration_dim: int=1,
        no_numerical_case_metadata:bool=False,
        **kwargs
    ) -> None:
        # super().__init__(numerical_metadata_dim, duration_embedding_dim, hidden_dim, activity_vocab_size, n_embd, n_head, n_layer, block_size, pad_code)
        super().__init__()
        # Setting the metadata embedding dimension, need to add 1 if odd (so it adds up)
        self.no_numerical_case_metadata=no_numerical_case_metadata
        if no_numerical_case_metadata:
            self.text_metadata_embedder = TextMetaDataEmbeddingModel(n_embd)
        else:
            self.text_metadata_embedder = TextMetaDataEmbeddingModel(n_embd//2)
            self.numerical_embedder = nn.Sequential(
                nn.Linear(numerical_metadata_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_embd//2)
            )

        # self.LanguageModel = LanguageDurationModel(activity_vocab_size, duration_embedding_dim, n_embd, n_head, n_layer, block_size, pad_code, duration_dim=duration_dim)
        self.LanguageModel = LanguageDurationModel(activity_vocab_size, duration_embedding_dim, n_embd, n_head, n_layer, block_size, pad_code, duration_dim=duration_dim)

    def forward(self,activity_sequence: torch.Tensor, metadata_text: str, metadata_numbers: torch.Tensor, durations: torch.Tensor):
        text_embedding = self.text_metadata_embedder(metadata_text)
        if not self.no_numerical_case_metadata:
            numerical_embedding = self.numerical_embedder(metadata_numbers)
            metadata_embedding = torch.cat((text_embedding, numerical_embedding), -1)
        else:
            metadata_embedding = text_embedding
        return self.LanguageModel(activity_sequence, metadata_embedding, durations)

class SOLMformer(MetadataSequenceModel):
    def __init__(
        self, 
        numerical_metadata_dim: int,
        duration_embedding_dim: int,
        hidden_dim: int,
        activity_vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        pad_code: int,
        event_numerical_metadata_dim: int=1,
        no_numerical_case_metadata=False,
        **kwargs
    ) -> None:
        # super().__init__(numerical_metadata_dim, duration_embedding_dim, hidden_dim, activity_vocab_size, n_embd, n_head, n_layer, block_size, pad_code)
        super().__init__(numerical_metadata_dim, hidden_dim, activity_vocab_size, n_embd, n_head, n_layer, block_size)
        # Setting the metadata embedding dimension, need to add 1 if odd (so it adds up)
        self.no_numerical_case_metadata=no_numerical_case_metadata
        if no_numerical_case_metadata:
            self.text_metadata_embedder = TextMetaDataEmbeddingModel(n_embd)
        else:
            self.text_metadata_embedder = TextMetaDataEmbeddingModel(n_embd//2)
            self.numerical_embedder = nn.Sequential(
                nn.Linear(numerical_metadata_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_embd//2)
            )

        self.event_numerical_embedder = nn.Sequential(
            nn.Linear(event_numerical_metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embd//2)
        )
        self.event_text_embedder = TextMetaDataEmbeddingModel(n_embd//2)
        self.LanguageModel = EventMetadataLanguageModel(
            vocab_size=activity_vocab_size, 
            event_embd=duration_embedding_dim, 
            n_embd=n_embd, 
            n_head=n_head, 
            n_layer=n_layer, 
            block_size=block_size
            )

    def forward(self, activity_sequence: torch.Tensor, metadata_text: torch.Tensor, metadata_numbers: torch.Tensor, event_metadata_numbers: torch.Tensor, event_metadata_text: torch.Tensor):
        B, T = activity_sequence.shape
        text_embedding = self.text_metadata_embedder(metadata_text)
        if not self.no_numerical_case_metadata:
            numerical_embedding = self.numerical_embedder(metadata_numbers)
            metadata_embedding = torch.cat((text_embedding, numerical_embedding), -1)
        else:
            metadata_embedding = text_embedding
        event_text_embedding = self.event_text_embedder(event_metadata_text)
        # Adding 1 to the initial shape to account for the slot for the case metadata token.
        event_text_embedding = event_text_embedding.reshape(B, T + 1, -1)
        event_numerical_embedding = self.event_numerical_embedder(event_metadata_numbers)
        event_metadata_embedding = torch.cat((event_text_embedding, event_numerical_embedding), -1)
        return self.LanguageModel(activity_sequence, metadata_embedding, event_metadata_embedding)
