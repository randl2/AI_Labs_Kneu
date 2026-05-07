import json
import numpy as np
from sklearn import covariance, cluster

input_file = 'company_symbol_mapping.json'
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

X = np.random.randn(100, len(symbols)) 
X /= X.std(axis=0)

edge_model = covariance.GraphicalLassoCV(max_iter=500) 
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print("\nРЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ ФОНДОВОГО РИНКУ:")
for i in range(num_labels + 1):
    cluster_names = names[labels == i]
    if len(cluster_names) > 0:
        print(f"Cluster {i+1} ==> {', '.join(cluster_names)}")
