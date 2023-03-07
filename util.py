#### 可复用的模块


import dgl

#### user item, 
#### 1. 从triple_set.tsv中取边
#### 2. 使用dgl构建成graph对象
def construst_kg():
    g = dgl.graph()
    return g