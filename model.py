import sys
import torch
import torch.nn.functional as F
import random
import numpy as np
import copy
from aggregator import Aggregator

class KGCN(torch.nn.Module):
    def __init__(self, num_drug, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_drug = num_drug
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)
        
        self._gen_adj()
            
        self.protein = torch.nn.Embedding(num_drug, args.dim)
        self.ent = torch.nn.Embedding(num_ent, args.dim)
        self.rel = torch.nn.Embedding(num_rel, args.dim)
        # self.sim_score = torch.nn.Embedding(num_drug, 2*args.dim)
        self.fc1 = torch.nn.Linear(2*args.dim + 6*num_drug, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        
    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent, self.n_neighbor, dtype=torch.long)
        
        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)
                
            self.adj_ent[e] = torch.LongTensor([ent for _, ent in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for rel, _ in neighbors])
        
    def forward(self, d1, d2, d1_sim, d2_sim, t1_sim, t2_sim, e1_sim, e2_sim):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = d1.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        d1 = d1.view((-1, 1))
        d2 = d2.view((-1, 1))

        # [batch_size, dim]
        d1_emb = self.ent(d1).squeeze(dim = 1)
        d1_entities, d1_relations = self._get_neighbors(d1)
        d1_item_embeddings = self._aggregate(d1_emb, d1_entities, d1_relations)
        # d1_scores = (d1_emb * d1_item_embeddings).sum(dim = 1)
        D1Emb = d1_emb * d1_item_embeddings
        D1Emb = torch.cat((D1Emb, d1_sim, t1_sim, e1_sim), 1)

        d2_emb = self.ent(d2).squeeze(dim=1)
        d2_entities, d2_relations = self._get_neighbors(d2)
        d2_item_embeddings = self._aggregate(d2_emb, d2_entities, d2_relations)
        # d2_scores = (d2_emb * d2_item_embeddings).sum(dim=1)
        D2Emb = d2_emb * d2_item_embeddings
        D2Emb = torch.cat((D2Emb, d2_sim, t2_sim, e2_sim), 1)
        d1d2 = torch.cat((D1Emb, D2Emb), 1)
        fc1 = self.fc1(d1d2.float())
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        out = torch.sigmoid(fc3)

            
        return out
    
    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []
        
        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
            
        return entities, relations
    
    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.ent(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]
        
        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid
            
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        
        return entity_vectors[0].view((self.batch_size, self.dim))