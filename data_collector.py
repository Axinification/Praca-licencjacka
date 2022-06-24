"""
Function for saving data from coinbase to csv
The library I use is located under:
https://github.com/David-Woroniuk/Historic_Crypto
As far as my research goes,
this is the most used and maintained library for gathering such information
"""
from Historic_Crypto import HistoricalData
import pandas as pd

from global_vars import CRYPTO, CURRENCY, DATE_START, DATE_END, ROOT_PATH

historicals = HistoricalData(CRYPTO+'-'+CURRENCY,
                             60, DATE_START, DATE_END).retrieve_data()

historicals.to_csv(ROOT_PATH)

input_df = pd.read_csv(ROOT_PATH)

input_df.tail()
