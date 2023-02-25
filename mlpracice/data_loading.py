#import required packages
import os
import requests
import pandas as pd


#Required constants
data_url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?date_received_max=2022-12-15&date_received_min=2022-12-01&field=all&format=json"
"""os.mkdir("data-downloads")
webfile = requests.get(data_url, allow_redirects =True)
open("complaints-2022-11-24_10_59.json", "wb").write(webfile.content)
"""
web_data = pd.read_json(data_url)
dict2df=pd.DataFrame.from_records(web_data["_source"])





"""data_url = f"https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/" \
                      f"?date_received_max=<todate>&date_received_min=<fromdate>" \
                      f"&field=all&format=json"
#required classes"""

