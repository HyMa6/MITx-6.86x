import pandas as pd

reviews_train = pd.read_csv("reviews_train.tsv", sep='\t', encoding ='unicode_escape')
print (reviews_train)



