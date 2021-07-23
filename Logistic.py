import dvc.api
import pandas as pd
import numpy as np

path='data/AdSmartABdata.csv'
repo='D:/Stella/Documents/10_Academy/Week-2/A_B_testing'
version='v1'

data_url=dvc.api.get_url(path=path,
                         repo=repo,
                         rev=version)


if __name__ == "__main__":
    warnings.filterwarningd("ignore")
    np.random.seed(40)

    data=pd.read_csv(data_url,sep=",")

