from transformers import BertForMaskedLM, BertTokenizer

model = BertForMaskedLM.from_pretrained('./trained_bert_masked_lm_model')
tokenizer = BertTokenizer.from_pretrained('./trained_bert_masked_lm_model')

def predict_masked_word(text):

    tokenized_text = tokenizer.encode_plus(text, return_tensors="pt")

    mask_token_index = torch.where(tokenized_text['input_ids'] == tokenizer.mask_token_id)
    mask_token_logits = model(**tokenized_text).logits
    mask_token_logits = mask_token_logits[0, mask_token_index.item(), :]

    top_k = 5 
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()


    predicted_tokens = [tokenizer.decode([token]) for token in top_k_tokens]

    return predicted_tokens

input_text = "The cat sat on the [MASK]."

predicted_words = predict_masked_word(input_text)
print("Predicted words:", predicted_words)
