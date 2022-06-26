import torch


def generate_bertlets_input(input_ids1, token_type_ids1, attention_mask1, input_ids2, token_type_ids2, attention_mask2):
    input_ids = [101] + input_ids1 + [102] + input_ids2 + [102]
    token_type_ids = [0] + token_type_ids1 + [0] + token_type_ids2 + [1]
    attention_mask = [1] + attention_mask1 + [1] + attention_mask2 + [1]
    if len(input_ids) > 512:
        input_ids = input_ids[:510] + [102]
    if len(token_type_ids) > 512:
        token_type_ids = token_type_ids[:510] + [1]
    if len(attention_mask) > 512:
        attention_mask = attention_mask[:510] + [1]
    return torch.LongTensor([[input_ids, token_type_ids, attention_mask]])

def calc_single_bertlets_score(input, bertlets_model, device):
    input = input.to(device)
    return bertlets_model.forward_once(input)