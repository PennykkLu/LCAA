import numpy as np


def metric(scores, targets, k=20):
    sub_scores = scores.topk(k)[1].cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    res = []
    for score,target in zip(sub_scores,targets):
        hit = float(np.isin(target, score))
        if len(np.where(score == target)[0]) == 0:
            mrr = 0
            ndcg = 0
        else:
            rank = np.where(score == target)[0][0] + 1
            mrr = 1 / rank
            ndcg = 1 / np.log2(rank + 1)
        res.append([hit,mrr,ndcg])

    return res
