#### 1. 加载数据：样本数据(dataloader), 知识图谱(dgl graph)
#### 2. 定义一下模型，传一些模型需要的参数，比如GCN模型需要图节点数目，定义优化器
#### 3. 训练模型
#### 4. 评估模型
#### 5. 节点的embedding保存起来
from config import *  ### 定义的参数
from model import GCN,GAT  #### 模型
# load_('') ### 加载镜像
df=pd.read_csv('train_data_csv',columns=['user_id', 'item_id', 'lable'])

### 这里加在triple.tsv user id item id, relation
g=dgl.graph((df['user_id'],df['item_id']))
g=dgl.ndata['feat']=torch.tensor((df['user_id'],df['item_id']))
### 1. 对现有特征做一些优化，特征工程，特征组合这些。2. 特征引入，引入新的特征
### 数据的优化工作收益一般大于模型优化

### 2. bad case, 优化的重点方向
g.edata['label']=torch.tensor(df['label'].values)
model = GAT(in_feats, hid_feats, num_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fcn = nn.CrossEntropyLoss()
for epoch in range(200):
    logits=model(g)
    labels=g.edata['label'].values
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

#### 存储模型

