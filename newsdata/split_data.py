import pandas as pd

data = pd.read_csv("./all-data.csv", encoding='latin-1').sample(frac=1).drop_duplicates()

data = data[['Label', 'Headline']].rename(columns={"Label":"label", "Headline":"text"})
data['label'] = data['label'].replace(['positive', 'neutral', 'negative'], [1,0,-1])

 
data['label'] = '__label__' + data['label'].astype(str)
data.iloc[0:int(len(data)*0.8)].to_csv('train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('test.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('dev.csv', sep='\t', index = False, header = False)