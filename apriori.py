#Lista de transações com produtos

transactions = [
    ['curd', 'sour cream'],['curd', 'orange', 'sour cream'], ['bread', 'cheese', 'butter'], ['bread', 'butter'], ['bread', 'milk'], 
    ['apple', 'orange', 'pear'], ['bread', 'milk', 'eggs'], ['tea', 'lemon'], ['curd', 'sour cream', 'apple'], ['eggs', 'wheat flour', 'milk'],
    ['pasta', 'cheese'], ['bread', 'cheese'], ['pasta', 'olive oil', 'cheese'], ['curd', 'jam'], ['bread', 'cheese', 'butter'],
    ['bread', 'sour cream', 'butter'], ['strawberry', 'sour cream'], ['curd', 'sour cream'], ['bread', 'coffee'], ['onion', 'garlic']
]

# Transformar os dados em array booleano via codificação one-hot

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)
df_itemsets = pd.DataFrame(encoded_array, columns=encoder.columns_)

print(df_itemsets)

# Visualizar o numero de transações e o numero de itens

print('Number of transactions: ', len(transactions))
print('Number of unique items: ', len(set(sum(transactions, []))))

#Identificando conjuntos de itens frequentes copm a função apriori()

from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(df_itemsets, min_support=0.1, use_colnames=True)

print(frequent_itemsets)

#V isualizar os itens com multiplos itens com lenght

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda itemset: len(itemset))
print(frequent_itemsets[frequent_itemsets['length'] >= 2])

Gerando regras de associação com association_rules()

from mlxtend.frequent_patterns import association_rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print(rules.iloc[:,0:7])

#Visualizando regras de associaçao com mapa de calor visualizando o lift

rules_plot = pd.DataFrame()
rules_plot['antecedents']= rules['antecedents'].apply(lambda x: ','.join(list(x)))
rules_plot['consequents']= rules['consequents'].apply(lambda x: ','.join(list(x)))
rules_plot['lift']= rules['lift'].apply(lambda x: round(x, 2))

# Transformando o DataFrame em uma matriz recorrendo ao metodo pivot()

pivot = rules_plot.pivot(index= 'antecedents', columns= 'consequents', values= 'lift')
print(pivot)

#Temos todas as variaveis para criar o mapa de calor, agora vamos separalas pra criar os eixos

antecedents = list(pivot.index.values)
consequents = list(pivot.columns)
import numpy as np
pivot = pivot.to_numpy()

# Agora, temos os rotulos do eixo y na lista antecedents, os rotulos do eixo x na lista consequents e os valores para o grafico no array do Numpy pivot. Usaremos todos esses componentes a seguir para criar um mapa de calor com Matplotlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
im = ax.imshow(pivot, cmap = 'Blues')
ax.set_xticks(np.arange(len(consequents)))
ax.set_yticks(np.arange(len(antecedents)))
ax.set_xticklabels(consequents)
ax.set_yticklabels(antecedents)
plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode="anchor")
for i in range(len(antecedents)):
  for j in range(len(consequents)):
    if not np.isnan(pivot[i, j]):
      test = ax.text(j, i, pivot[i, j], ha="center", va="center")
ax.set_title("Lift metric for frequent itemsets")
fig.tight_layout()
plt.show()
