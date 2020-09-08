import json
import testbed_utils as utils

# parsed = json.loads('meta_data.json')
with open('meta_data.json', 'r') as fp:
			meta_data_ = json.load(fp)
scaler = utils.dataframescaler(meta_data_)