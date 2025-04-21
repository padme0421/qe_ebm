from transformers import AutoTokenizer
mbart_tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
mbart_vocab = mbart_tokenizer.get_vocab()
mbart_token_set = set(mbart_vocab.keys())
print("number of tokens in mbart vocab: ", len(mbart_token_set))

print(mbart_tokenizer.pad_token_id)
print(mbart_tokenizer.eos_token_id)
print(mbart_tokenizer.bos_token_id)

xlm_tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-large")
xlm_vocab = xlm_tokenizer.get_vocab()
xlm_token_set = set(xlm_vocab.keys())
print("number of tokens in xlm vocab: ", len(xlm_token_set))
print(xlm_tokenizer.pad_token_id)
print(xlm_tokenizer.eos_token_id)
print(xlm_tokenizer.bos_token_id)

nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_vocab = nllb_tokenizer.get_vocab()
nllb_token_set = set(nllb_vocab.keys())
print("number of tokens in nllb vocab: ", len(nllb_token_set))
print(nllb_tokenizer.pad_token_id)
print(nllb_tokenizer.eos_token_id)
print(nllb_tokenizer.bos_token_id)

joint_tokens = mbart_token_set.intersection(nllb_token_set)
print("number of joint tokens: ", len(joint_tokens))

print("compare index of joint tokens")
index_mapping = {}
for token in joint_tokens:
    index_mapping[token] = (mbart_vocab[token], nllb_vocab[token])

samples = 10
for k,v in index_mapping.items():
    if samples == 0:
        break
    print(k + ": " + str(v[0]) + ' ' + str(v[1]))
    samples -=1
