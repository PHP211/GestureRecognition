import json

keys = [0, 'aaaaa']

# Lưu vào file JSON
with open('data.json', 'w') as f:
    json.dump(keys, f)