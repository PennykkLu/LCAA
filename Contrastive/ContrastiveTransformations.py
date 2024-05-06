import torch

class ContrastiveTransformations:
    """
    generate anchor-positive pairs
    """
    def __init__(self,transforms_book):
        self.transforms_book = transforms_book   #key: query item   value: [items,scores] sorted by correlation scores in ascending order

    def __call__(self, sess_batch,mul_len,n_node):
        batch_size,dim = sess_batch.shape
        new_sess_batch = torch.zeros(batch_size, dim).long()
        for i, session in enumerate(sess_batch):
            org_len = int(torch.count_nonzero(session))
            new_len = min(mul_len + org_len, dim)
            first_item = int(session[0])
            if new_len != org_len and first_item in self.transforms_book.keys():
                cad_items = list(self.transforms_book[first_item][0])
                if len(cad_items) < new_len - org_len:
                    Increased_items = cad_items[:-1]
                else:
                    Increased_items = cad_items[-(new_len - org_len + 1):-1]
                new_session = Increased_items + session[:org_len].tolist()
                new_session = new_session + [0] * max(dim - len(new_session), 0)
                new_sess_batch[i, :] = torch.LongTensor(new_session)
            else:
                new_sess_batch[i, :] = session
        new_lens_batch = torch.count_nonzero(new_sess_batch, dim=1)
        return new_sess_batch,new_lens_batch