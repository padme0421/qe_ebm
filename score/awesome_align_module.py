import torch
import itertools

# for MBERT
from awesome_align import modeling
from awesome_align.configuration_bert import BertConfig
from awesome_align.modeling import BertForMaskedLM
from awesome_align.tokenization_bert import BertTokenizer

# for XLMR
from awesome_align import modeling
from awesome_align.configuration_xlmr import XLMRobertaConfig
from awesome_align.modeling_xlmr import XLMRobertaForMaskedLM
from awesome_align.tokenization_xlmr import XLMRobertaTokenizer

class AwesomeAligner:
    def __init__(self, model_name_or_path='bert-base-multilingual-cased', device=None):
        config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer
        config = config_class.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        modeling.PAD_ID = tokenizer.pad_token_id
        modeling.CLS_ID = tokenizer.cls_token_id
        modeling.SEP_ID = tokenizer.sep_token_id
        self.model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config
        )
        self.model.requires_grad_(False)
        
        self.tokenizer = tokenizer
        #if device == None:
        #    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #else:
        #    self.device = device

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def align(self, line, output_prob = False):
        src, tgt = line.split(' ||| ')
        assert src.rstrip() != '' and tgt.rstrip() != ''

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=self.tokenizer.max_len)['input_ids']
        assert len(ids_src[0]) != 2 or len(ids_tgt[0]) != 2
        
        #ids_src = ids_src.to(self.device)
        #ids_tgt = ids_tgt.to(self.device)
        input_device = next(self.model.parameters()).device

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]

        #print(ids_src.device)
        #print(ids_tgt.device)
        word_aligns = self.model.get_aligned_word(ids_src, ids_tgt, [bpe2word_map_src], [bpe2word_map_tgt], input_device, 0, 0, test=True, output_prob=output_prob)[0]

        return word_aligns

#model = AwesomeAligner()
#print(model.align('order , please .   ||| a le ordre .'))

class AwesomeAlignerXLMR():
    def __init__(self, model_name_or_path='microsoft/infoxlm-large', device=None):
        config_class, model_class, tokenizer_class = XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer
        config = config_class.from_pretrained(model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        modeling.PAD_ID = tokenizer.pad_token_id
        modeling.CLS_ID = tokenizer.cls_token_id
        modeling.SEP_ID = tokenizer.sep_token_id
        self.model: XLMRobertaForMaskedLM = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config
        )
        self.model.requires_grad_(True)
        
        self.tokenizer = tokenizer
        #if device == None:
        #    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #else:
        #    self.device = device

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def align(self, line, output_prob = False):
        src, tgt = line.split(' ||| ')
        assert src.rstrip() != '' and tgt.rstrip() != ''

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', max_length=self.tokenizer.max_len, truncation=True)['input_ids'], self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', max_length=self.tokenizer.max_len, truncation=True)['input_ids']
        assert len(ids_src[0]) != 2 or len(ids_tgt[0]) != 2
        
        input_device = next(self.model.parameters()).device

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]


        word_aligns = self.model.get_aligned_word(ids_src, ids_tgt, [bpe2word_map_src], [bpe2word_map_tgt], input_device, 0, 0, test=True, output_prob=output_prob)[0]

        return word_aligns