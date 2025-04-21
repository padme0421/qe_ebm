from transformers import MBartForConditionalGeneration
from transformers import AutoTokenizer
import torch
# without any pretraining, mbart-large-50 can't produce translation
# mbart-large-50-many-to-many-mmt can

mbart50_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50",src_lang="en_XX", tgt_lang="ko_KR")
print(tokenizer.tgt_lang)

example_english_phrase = "I am cute"
#ref_korean = "저는 귀엽습니다"
inputs = tokenizer(example_english_phrase, max_length=10, padding=True, truncation=True, return_tensors="pt")
                   #text_target=ref_korean, max_length=10, padding=True, truncation=True, return_tensors="pt")
print("inputs: ", inputs)

# generation
generated_ids = mbart50_model.generate_with_grad(**inputs, num_beams=4, 
                                       max_length=10, 
                                       forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])
print("translation ids: ",generated_ids)
ko_translation = tokenizer.batch_decode(generated_ids)
print("translation: ", ko_translation)

# one forward pass
'''
outputs = mbart50_model(**inputs)
preds = torch.argmax(outputs.logits, dim=-1)
print("preds: ", preds)
one_forward_pass = tokenizer.batch_decode(preds)
print("one forward pass: ", one_forward_pass)
'''
# 

'''
encoder = mbart50_model.get_encoder()
encoder_outputs = encoder(**inputs)
decoder_start_tokens = torch.ones_like(inputs.input_ids)[:, :1] * mbart50_model.config.decoder_start_token_id
decoder_forced_start_tokens = torch.ones_like(inputs.input_ids)[:, :1] * tokenizer.lang_code_to_id['de_DE']
decoder_start_tokens = torch.cat((decoder_start_tokens, decoder_forced_start_tokens), dim=1)
            
model_kwargs = {"encoder_outputs": encoder_outputs}
unsup_outputs = mbart50_model.greedy_search(decoder_start_tokens, max_length = 10, 
                                            pad_token_id = mbart50_model.config.pad_token_id, eos_token_id = mbart50_model.config.eos_token_id,
                                            output_attentions=True, output_hidden_states=True, output_scores=True,
                                            return_dict_in_generate=True,
                                            **model_kwargs,
                )
print(unsup_outputs)
output_sent = tokenizer.batch_decode(unsup_outputs.sequences)
print("output sent: ", output_sent)
'''