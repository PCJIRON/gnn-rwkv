import json
with open('vocab.json', 'r') as f:
    data = json.load(f)
print("Keys in vocab.json:", data.keys())

if 'w2i' in data:
    print("w2i size:", len(data['w2i']))

if 'i2w' in data:
    print("i2w size:", len(data['i2w']))
    first_key = list(data['i2w'].keys())[0]
    try:
        int(first_key)
        print("i2w seems correct.")
    except ValueError:
        print("i2w is inverted. Fixing...")
        data['i2w'] = {str(v): k for k, v in data['w2i'].items()}
        with open('vocab.json', 'w') as f:
            json.dump(data, f)
        print("i2w fixed and saved.")
else:
    print("i2w NOT found. Reconstructing...")
    data['i2w'] = {str(v): k for k, v in data['w2i'].items()}
    with open('vocab.json', 'w') as f:
        json.dump(data, f)
    print("i2w reconstructed and saved.")
