# import file for training
text = "Hello my name is Charlotte. What is your name? My name is Joe. How are you today? I am good. How are you? I am doing well."

chars = sorted(list(set(text)))

str_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_str = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [str_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_str[i] for i in l])

encoded_text = encode(text)

train = encoded_text[:int(0.8*len(encoded_text))]
validate = encoded_text[int(0.8*len(encoded_text)):int(0.9*len(encoded_text))]
test = encoded_text[int(0.9*len(text)):]

block_size = 4
print(train[:block_size+1])

batch_size = 4