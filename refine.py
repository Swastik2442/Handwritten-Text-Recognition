import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

import nltk
nltk.download("punkt_tab")
from nltk.tokenize import word_tokenize

class TextRefiner:
    """Uses a Pre-trained Masked Language Model to refine Text"""

    def __init__(self, model_id="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval()

    def refine_text(self, sentence):
        words = word_tokenize(sentence)
        refined = []
        for i, word in enumerate(words):
            temp_words = words.copy()
            temp_words[i] = "[MASK]"
            masked_sent = " ".join(temp_words)
            inputs = self.tokenizer(masked_sent, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            mask_token_id = self.tokenizer.mask_token_id
            mask_token_index = (inputs.input_ids == mask_token_id)[0].nonzero(as_tuple=True)[0]
            predicted_id = logits[0, mask_token_index].argmax(dim=-1)
            predicted_token = self.tokenizer.decode(predicted_id)

            # Replace only if BERT gives a different valid suggestion
            if predicted_token.lower() != word.lower() and predicted_token.isalpha():
                refined.append(predicted_token)
            else:
                refined.append(word)
        return " ".join(refined)
