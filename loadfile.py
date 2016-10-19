import requests
import json
import StringIO
import pandas

def _get_id(persistent_id, idx, server):
    print server
    server = "dataverse.harvard.edu"
    url = "http://%s/api/datasets/:persistentId/?persistentId=%s" % (server,
        persistent_id)
    json_content = json.loads(requests.get(url).text)
    id_ = json_content["data"]["latestVersion"]["files"][idx]["dataFile"]["id"]
    return id_

def load_data(source):
    params = {"index_col": 0, "skipinitialspace": True}
    if source.isdigit():
        idx = int(source)-1
        file_content = _get_online_file(idx)
        f = StringIO.StringIO(file_content)
        df = pandas.read_table(f, low_memory=False, **params)
    else:
        df = pandas.read_csv(source, **params)
 #   print df.head()
 #   print df.dtypes
    return df


def _get_online_file(idx):
    server = "dataverse.harvard.edu"
    persistent_id = "doi:10.7910/DVN/YHNI6O"
    id_ = _get_id(persistent_id, idx, server=server)
    url = "https://%s/api/access/datafile/%s" % (server, id_)
    data = requests.get(url)
    return data.text

if __name__ == "__main__":
    pass
