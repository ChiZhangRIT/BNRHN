''' 
The downloaded MSCOCO annotation lacks of key "type" for some reason.
This script is doing nothing but add "type" as a key in the dictionary.
'''

import json

in_file = '/cis/phd/cxz2081/data/mscoco/captioning/annotations/captions_val2014.json'
out_file = '/cis/phd/cxz2081/data/mscoco/captioning/annotations/captions_val2014_addtype.json'

with open(in_file, 'r') as f:
  data = json.load(f)

data['type'] = u'captions'

with open(out_file,'w') as f:
  json.dump(data, f)
