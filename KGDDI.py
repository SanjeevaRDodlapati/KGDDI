# import pandas as pd
import numpy as np
import argparse
# import random
from model import KGCN
from data_loader import DataLoader
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


# prepare arguments (hyperparameters)
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='drugs', help='which dataset to use')
parser.add_argument('--aggregator', type=str, default='sum', help='which aggregator to use')
parser.add_argument('--n_epochs', type=int, default=3, help='the number of epochs')
parser.add_argument('--n_sample', type=int, default=3233647, help='the number of samples to use')
parser.add_argument('--neighbor_sample_size', type=int, default=8, help='the number of neighbors to be sampled')
parser.add_argument('--dim', type=int, default=16, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations when computing entity representation')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--ratio', type=float, default=0.7, help='size of training dataset')

args = parser.parse_args(['--l2_weight', '1e-4'])


# build dataset and knowledge graph
data_loader = DataLoader(args.dataset)
kg = data_loader.load_kg()
df_dataset = data_loader.load_dataset(args.n_sample)
df_kg = data_loader.df_kg
drug_sim = data_loader.drug_sim
target_sim = data_loader.target_sim
enzyme_sim = data_loader.enzyme_sim


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df, drug1, drug2):
        self.df = df
        self.drug1 = drug1
        self.drug2 = drug2
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['drug1_enc'])
        item_id = np.array(self.df.iloc[idx]['drug2_enc'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label

def sim_scores(data_loader, drug_enc):
    drug_sim_score = []
    drug_enc = np.array(drug_enc)
    for d in drug_enc:
        temp = data_loader.drug_sim[data_loader.drug_sim.drug_id==d]
        temp = temp.iloc[:, 1:].values
        temp = temp.reshape(temp.shape[1])
        drug_sim_score.append(temp)

    drug_sim_score  = np.array(drug_sim_score)
    drug_sim_score = torch.from_numpy(drug_sim_score)
    return drug_sim_score




# Dataset class
class DDIDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        drug1_enc = np.array(self.df.iloc[idx]['drug1_enc'])
        drug2_enc = np.array(self.df.iloc[idx]['drug2_enc'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return drug1_enc, drug2_enc, label



# train test split
x_train, x_test, y_train, y_test = train_test_split(df_dataset, df_dataset['label'], test_size=1 - args.ratio, shuffle=True, random_state=999)
train_dataset = DDIDataset(x_train)
test_dataset = DDIDataset(x_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)


# prepare network, loss function, optimizer
num_drug, num_entity, num_relation = data_loader.get_num()
drug_encoder, entity_encoder, relation_encoder = data_loader.get_encoders()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = KGCN(num_drug, num_entity, num_relation, kg, args, device).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
print('device: ', device)

# train
loss_list = []
test_loss_list = []
auc_score_list = []

for epoch in range(args.n_epochs):
    running_loss = 0.0
    for i, (drug1_ids, drug2_ids, labels) in enumerate(train_loader):
        # Get Proten and drug from kg data fro each drug in DDIDataset
        # protein_ids, item_ids =  KGCNDataset(df_kg, drug1_ids, drug2_ids)# Get it from df_kg data
        drug1_sim_score = sim_scores(data_loader, drug1_ids)
        drug2_sim_score = sim_scores(data_loader, drug2_ids)
        drug1_sim_score, drug2_sim_score = drug1_sim_score.to(device), drug2_sim_score.to(device)

        target1_sim_score = sim_scores(data_loader, drug1_ids)
        target2_sim_score = sim_scores(data_loader, drug2_ids)
        target1_sim_score, target2_sim_score = target1_sim_score.to(device), target2_sim_score.to(device)

        enzyme1_sim_score = sim_scores(data_loader, drug1_ids)
        enzyme2_sim_score = sim_scores(data_loader, drug2_ids)
        enzyme1_sim_score, enzyme2_sim_score = enzyme1_sim_score.to(device), enzyme2_sim_score.to(device)

        drug1_ids, drug2_ids, labels = drug1_ids.to(device), drug2_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(drug1_ids, drug2_ids, drug1_sim_score, drug2_sim_score, target1_sim_score, target2_sim_score, enzyme1_sim_score, enzyme2_sim_score)
        outputs = outputs.reshape(labels.shape[0])
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i%10==0:
            print(f'step: {i+1} , average loss: {running_loss/(i+1)}')

    # print train loss per every epoch
    print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
    loss_list.append(running_loss / len(train_loader))

    # evaluate per every epoch
    with torch.no_grad():
        test_loss = 0
        total_roc = 0
        for drug1_ids, drug2_id_ids, labels in test_loader:
            # user_ids, item_ids, labels = user_ids.to(device), item_ids.to(device), labels.to(device)
            # outputs = net(user_ids, item_ids)
            drug1_sim_score = sim_scores(data_loader, drug1_ids)
            drug2_sim_score = sim_scores(data_loader, drug2_ids)
            drug1_sim_score, drug2_sim_score = drug1_sim_score.to(device), drug2_sim_score.to(device)

            target1_sim_score = sim_scores(data_loader, drug1_ids)
            target2_sim_score = sim_scores(data_loader, drug2_ids)
            target1_sim_score, target2_sim_score = target1_sim_score.to(device), target2_sim_score.to(device)

            enzyme1_sim_score = sim_scores(data_loader, drug1_ids)
            enzyme2_sim_score = sim_scores(data_loader, drug2_ids)
            enzyme1_sim_score, enzyme2_sim_score = enzyme1_sim_score.to(device), enzyme2_sim_score.to(device)

            drug1_ids, drug2_ids, labels = drug1_ids.to(device), drug2_ids.to(device), labels.to(device)
            # optimizer.zero_grad()
            outputs = net(drug1_ids, drug2_ids, drug1_sim_score, drug2_sim_score, target1_sim_score, target2_sim_score,
                          enzyme1_sim_score, enzyme2_sim_score)
            outputs = outputs.reshape(labels.shape[0])
            # outputs = outputs.reshape(labels.shape[0])
            test_loss += criterion(outputs, labels).item()
            total_roc += roc_auc_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy())
        print('[Epoch {}]test_loss: '.format(epoch + 1), test_loss / len(test_loader))
        test_loss_list.append(test_loss / len(test_loader))
        print('[Epoch {}]test_AUC: '.format(epoch + 1), total_roc / len(test_loader))
        auc_score_list.append(total_roc / len(test_loader))

print('Done!')


#%%
# plot losses / scores
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))  # 1 row, 2 columns
ax1.plot(loss_list)
ax1.plot(test_loss_list)
ax2.plot(auc_score_list)

plt.tight_layout()