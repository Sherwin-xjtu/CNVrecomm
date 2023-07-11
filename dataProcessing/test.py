import pandas as pd
import numpy as np
data = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
print(data)


data.insert(0, 'd', 0)
print(data.shape[1])
print(data)
