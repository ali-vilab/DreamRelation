import torch
import torch.nn.functional as F


def calculate_relational_contrastive_loss(relation_embeds, positive_embeds, negative_embds, temperature=0.07):
    # batch_size, 1, dim -> batch_size, dim
    relation_embeds = relation_embeds.squeeze(1)
    
    # concatenate positive and negative embeddings along the second dimension
    pn_embeds = torch.cat([positive_embeds, negative_embds], dim=1)  # (batch_size, num_positive + num_negative, dim)
    
    # normalize embeddings
    relation_embeds_normalized = F.normalize(relation_embeds, p=2, dim=1)
    pn_embeds_normalized = F.normalize(pn_embeds, p=2, dim=2)
    
    # compute logits
    logits = torch.einsum('bd,bkd->bk', [relation_embeds_normalized, pn_embeds_normalized])
    
    # scale the logits by temperature
    logits /= temperature
    
    # split logits into positive and negative parts
    num_positive = positive_embeds.size(1)
    positive_logits = logits[:, :num_positive]  # (batch_size, num_positive)
    negative_logits = logits[:, num_positive:]  # (batch_size, num_negative)
    
    # compute the nominator and denominator for Multi-Instance InfoNCE loss
    nominator = torch.logsumexp(positive_logits, dim=1)  # (batch_size,)
    denominator = torch.logsumexp(logits, dim=1)  # (batch_size,)
    
    # compute mean loss across the batch
    loss = torch.mean(denominator - nominator)
    
    return loss