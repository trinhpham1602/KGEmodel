import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random
import model
device = "cuda" if torch.cuda.is_available() else "cpu"


class Discriminator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(emb_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, hrt_concat_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(hrt_concat_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.gen(x)


def run(model, pos_triples, neg_triples, emb_dim, lr, step, n_negs, k_negs, mode):
    disc = Discriminator(emb_dim).to(device)  # environment
    gen = Generator(3*emb_dim).to(device)  # agent
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    b = 0
    pos_triples = pos_triples.to(device)
    neg_triples = neg_triples.to(device)
    #start = time.time()
    for _ in range(step):
        opt_disc.zero_grad()
        opt_gen.zero_grad()
        # get embs for pos and neg
        # print(pos_triples)
        pos_embs = model.take_embs(pos_triples)
        h, r, t = model.take_embs((pos_triples, neg_triples), mode)
        global sum_negs, concat_hrt
        if mode == "head-batch":
            sum_negs = h + (r - t)
        if mode == "tail-batch":
            sum_negs = (h + r) - t
        loss_d = -torch.log(disc(pos_embs)) - \
            torch.sum(torch.log(1 - disc(sum_negs)))
        loss_d.mean().backward()
        opt_disc.step()
        reward = - disc(sum_negs).detach()
        concat_hrt = torch.empty((pos_embs.size(0), n_negs, emb_dim*3))
        if mode == "head-batch":
            for inx in range(h.size(1)):
                concat_hrt[:, inx] = torch.cat((h[:, inx], torch.cat(
                    (r, t), dim=1).view(pos_embs.size(0), emb_dim*2)), dim=1)
        if mode == "tail-batch":
            for inx in range(t.size(1)):
                concat_hrt[:, inx] = torch.cat((torch.cat((h, r), dim=1).view(
                    pos_embs.size(0), emb_dim*2), t[:, inx]), dim=1)
        concat_hrt = concat_hrt.detach().to(device)
        loss_g = (reward - b) * torch.log(gen(concat_hrt))
        loss_g.mean().backward()
        opt_gen.step()
        global high_neg_triples
        high_neg_triples = torch.empty(
            (pos_triples.size(0), k_negs), device=device)
        for inx, indices in enumerate(torch.topk(gen(concat_hrt).view(pos_triples.size(0), n_negs), k_negs, dim=1).indices):
            high_neg_triples[inx] = neg_triples[inx][indices]
        b = reward/pos_triples.size(0)
    #print(time.time() - start)
    return disc, high_neg_triples.long()
