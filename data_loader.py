import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random


class DataLoader:
    '''
    Data Loader class which makes dataset for training / knowledge graph dictionary
    '''
    def __init__(self, data):
        self.cfg = {
            'movie': {
                'item2id_path': 'data/movie/item_index2entity_id.txt',
                'kg_path': 'data/movie/kg.txt',
                'rating_path': 'data/movie/ratings.csv',
                'rating_sep': ',',
                'threshold': 4.0
            },
            'music': {
                'item2id_path': 'data/music/item_index2entity_id.txt',
                'kg_path': 'data/music/kg.txt',
                'rating_path': 'data/music/user_artists.dat',
                'rating_sep': '\t',
                'threshold': 0.0
            },
            'drugs': {
                'kg_path': 'data/kg_data.csv',
                'drug_sim_path': 'data/drug_similarity.csv',
                'target_sim_path': 'data/target_similarity.csv',
                'enzyme_sim_path': 'data/enzyme_similarity.csv',
                'DDI_all_path': 'data/df_DDI_final.csv'
            }
        }
        self.data = data
        

        df_kg = pd.read_csv(self.cfg[data]['kg_path']) #, sep='\t')#, header=None, names=['drug', 'entity', 'entity_id', 'relation'])
        # df_kg.rename(columns={'function': "relation"}, inplace=True)
        self.df_kg = df_kg

        self.drug_sim = pd.read_csv(self.cfg[data]['drug_sim_path'])
        self.drug_sim.rename(columns={'Unnamed: 0': 'drug_id'}, inplace=True)
        # self.drug_sim = self.drug_sim.set_index('drug_id')

        self.target_sim = pd.read_csv(self.cfg[data]['target_sim_path'])
        self.target_sim.rename(columns={'Unnamed: 0': 'drug_id'}, inplace=True)
        # self.target_sim = self.target_sim.set_index('drug_id')

        self.enzyme_sim = pd.read_csv(self.cfg[data]['enzyme_sim_path'])
        self.enzyme_sim.rename(columns={'Unnamed: 0': 'drug_id'}, inplace=True)
        # self.enzyme_sim = self.enzyme_sim.set_index('drug_id')

        self.DDI = pd.read_csv(self.cfg[data]['DDI_all_path'])

        
        self.drug_encoder = LabelEncoder()
        self.entity_encoder = LabelEncoder()
        self.relation_encoder = LabelEncoder()

        self._encoding()
        
    def _encoding(self):
        '''
        Fit each label encoder and encode knowledge graph
        '''
        # self.protein_encoder.fit(self.df_kg['protein_id'])
        self.drug_encoder.fit(self.df_kg['drugbank_id'])
        # df_udrug_index['id'] and df_kg[['drug', 'target']] represents new entity ID
        # self.entity_encoder.fit(pd.concat([self.df_udrug_index['index'], self.df_kg['drugbank_id'], self.df_kg['protein_id']]))

        self.entity_encoder.fit(
            pd.concat([self.df_kg['drugbank_id'], self.df_kg['protein_id']]))
        self.relation_encoder.fit(self.df_kg['function'])

        # self.entity_encoder.fit(
        #     pd.concat([self.df_kg['drugbank_id'], self.df_kg['protein_id'], self.df_kg['function']]))
        
        # encode df_kg
        self.df_kg['head'] = self.entity_encoder.transform(self.df_kg['drugbank_id'])
        self.df_kg['tail'] = self.entity_encoder.transform(self.df_kg['protein_id'])
        self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['function'])
        # self.df_kg['relation'] = self.relation_encoder.transform(self.df_kg['function'])
        self.df_kg = self.df_kg.loc[:, ['drugbank_id', 'protein_id', 'function', 'head', 'tail', 'relation']]
        self.drug_sim['drug_id'] = self.entity_encoder.transform(self.drug_sim['drug_id'])
        self.target_sim['drug_id'] = self.entity_encoder.transform(self.target_sim['drug_id'])
        self.enzyme_sim['drug_id'] = self.entity_encoder.transform(self.enzyme_sim['drug_id'])

    def _build_dataset(self, n_sample):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        # df_rating update
        df_dataset = pd.DataFrame()
        df_dataset['drug1'] = self.DDI['drug1']
        df_dataset['drug2'] = self.DDI['drug2']
        df_dataset['drug1_enc'] = self.entity_encoder.transform(self.DDI['drug1'])
        df_dataset['drug2_enc'] = self.entity_encoder.transform(self.DDI['drug2'])
        df_dataset['label'] = self.DDI['label']

        # df_dataset['userID'] = self.df_kg['protein_factorID']
        # # update to new id
        # # item2id_dict = dict(zip(self.df_item2id['item'], self.df_item2id['id']))
        # # self.df_rating['itemID'] = self.df_rating['itemID'].apply(lambda x: item2id_dict[x])
        # df_dataset['itemID'] = self.df_kg['drug_factorID']
        # df_dataset['drug_similarity'] = self.df_kg['drug_similarity']
        # df_dataset['target_similarity'] = self.df_kg['target_similarity']
        # df_dataset['enzyme_similarity'] = self.df_kg['enzyme_similarity']
        # df_dataset['label'] = self.df_kg['label']
        #
        # # negative sampling
        # df_dataset = df_dataset[df_dataset['label']==1]
        # # df_dataset requires columns to have new entity ID
        # full_item_set = set(range(len(self.entity_encoder.classes_)))
        # user_list = []
        # item_list = []
        # dsm_list = []
        # tsm_list = []
        # esm_list = []
        # label_list = []
        # for user, group in df_dataset.groupby(['userID']):
        #     item_set = set(group['itemID'])
        #     negative_set = full_item_set - item_set
        #     negative_sampled = random.sample(negative_set, len(item_set))
        #     user_list.extend([user] * len(negative_sampled))
        #     item_list.extend(negative_sampled)
        #     label_list.extend([0] * len(negative_sampled))
        # negative = pd.DataFrame({'userID': user_list, 'itemID': item_list, 'label': label_list})
        # df_dataset = pd.concat([df_dataset, negative])
        #
        # df_dataset = df_dataset.sample(frac=1, replace=False, random_state=999)
        df_dataset = df_dataset.sample(n=n_sample, replace=False, random_state=999)
        # df_dataset.reset_index(inplace=True, drop=True)
        print('Done')
        return df_dataset
        
        
    def _construct_kg(self):
        '''
        Construct knowledge graph
        Knowledge graph is dictionary form
        'head': [(relation, tail), ...]
        '''
        print('Construct knowledge graph ...', end=' ')
        kg = dict()
        for i in range(len(self.df_kg)):
            head = self.df_kg.iloc[i]['head']
            relation = self.df_kg.iloc[i]['relation']
            tail = self.df_kg.iloc[i]['tail']
            if head in kg:
                kg[head].append((relation, tail))
            else:
                kg[head] = [(relation, tail)]
            if tail in kg:
                kg[tail].append((relation, head))
            else:
                kg[tail] = [(relation, head)]
        print('Done')
        return kg
        
    def load_dataset(self, n_sample):
        return self._build_dataset(n_sample)

    def load_kg(self):
        return self._construct_kg()
    
    def get_encoders(self):
        return (self.drug_encoder, self.entity_encoder, self.relation_encoder)
    
    def get_num(self):
        return (len(self.drug_encoder.classes_), len(self.entity_encoder.classes_), len(self.relation_encoder.classes_))
