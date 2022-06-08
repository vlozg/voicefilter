import os
import pandas as pd
import networkx as nx
from pathlib import Path

if __name__ == '__main__':
    
    df = pd.read_csv('ZaloAI2020/private-test-label.csv')
    
    # Construct graph from edge list
    G = nx.from_pandas_edgelist(df[df['label']==1], 'audio_1', 'audio_2')
    
    sub = df[df['label']==0]
    G.add_nodes_from(sub["audio_1"])
    G.add_nodes_from(sub["audio_2"])
    
    # From graph to list
    groups =  list(nx.connected_components(G))
    map = dict()
    for i, group in enumerate(groups):
        for f in list(group):
            src=os.path.join('ZaloAI2020/private-test', f)
            dst=os.path.join('ZaloAI2020/private-test', str(i), f)
            map[f] = dst
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            Path(src).rename(dst) 
        
    df["audio_1"] = "datasets/" + df["audio_1"].map(map)
    df["audio_2"] = "datasets/" + df["audio_2"].map(map)
    df.to_csv("ZaloAI2020/private-test-fixed.csv", index=False)