import torch

def generate_att_mask_pos_ids(token_ids: torch.Tensor, pad_token_id: int):
    attention_mask: torch.Tensor = token_ids.masked_fill(token_ids == pad_token_id, 0).masked_fill(token_ids != pad_token_id, 1)
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return attention_mask, position_ids

if __name__ == '__main__':
    token_ids = torch.tensor([12, 31, 52, 52])
    print(generate_att_mask_pos_ids(token_ids, pad_token_id=52))
