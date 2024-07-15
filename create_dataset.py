import pandas as  pd
from datasets import load_dataset, Features, Image, ClassLabel
import random
random.seed(8883)
if __name__ == '__main__':
    df = pd.read_csv('./Human Action Recognition/Training_set.csv')
    df['filename'] = df.filename.apply(lambda x:'./Human Action Recognition/train/'+x)
    excluded_df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.2))
    val_df = excluded_df.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=0.5))
    test_df = excluded_df[~excluded_df.filename.isin(val_df.filename.values)]
    train_df = df[~df.filename.isin(excluded_df.filename.values)]
    train_df.to_csv('train.csv',index=False)
    val_df.to_csv('val.csv',index=False)
    test_df.to_csv('test.csv',index=False)
