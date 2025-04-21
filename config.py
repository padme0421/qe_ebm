ko_en_config = {
    "src": "korean", # dataset src column name
    "trg": "english", # dataset trg column name
    "src_tokenizer": "ko_core_news_sm",
    "trg_tokenizer": "en_core_web_sm",
    "dataset": "msarmi9/korean-english-multitarget-ted-talks-task"
}

de_en_config = {
    "src": "de",
    "trg": "en"
}

opus_en_ko_pl_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "opus100"
}


opus_en_ko_pl_mbart50_config_fine = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "alphahg/mbart-large-50-finetuned-en-to-ko-8603428",
    "dataset": "opus100"
}

opus_en_ko_pl_mbart50_mmt_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "opus100"
}

opus_en_ko_pl_mbartcc25_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "facebook/mbart-large-cc25",
    "dataset": "opus100"
}

iwslt17_en_de_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "de",
    "model_trg_code": "de_DE",
    "data_trg_code": "de_DE",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "iwslt2017"
}

robust_en_de_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "de",
    "model_trg_code": "de_DE",
    "data_trg_code": "de_DE",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "robust_wmt20"
}



iwslt17_en_de_mbart50_mmt_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "de",
    "model_trg_code": "de_DE",
    "data_trg_code": "de_DE",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "iwslt2017"
}

iwslt17_en_zh_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "zh",
    "model_trg_code": "zh_CN",
    "data_trg_code": "zh_CN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "iwslt2017"
}


iwslt17_en_ko_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "ko",
    "model_trg_code": "ko_KR",
    "data_trg_code": "ko_KR",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "iwslt2017",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "ko_core_news_md"
}

iwslt17_ko_en_mbart50_config = {
    "src": "ko",
    "model_src_code": "ko_KR",
    "data_src_code": "ko_KR",
    "trg": "en",
    "model_trg_code": "en_XX",
    "data_trg_code": "en_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "iwslt2017",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "ko_core_news_md"
}



iwslt17_en_ko_xlmr_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "xlm-roberta-base",
    "dataset": "iwslt2017",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "ko_core_news_md"
}

iwslt17_en_ko_mt5_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "google/mt5-small",
    "dataset": "iwslt2017",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "ko_core_news_md"
}

iwslt17_en_ko_bart_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ko",
    "trg_code": "ko_KR",
    "model_name_or_path": "facebook/bart-base",
    "dataset": "iwslt2017",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "ko_core_news_md"
}

iwslt17_en_ja_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "ja",
    "model_trg_code": "ja_XX",
    "data_trg_code": "ja_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "iwslt2017"
}

mtnt_en_ja_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX", # not needed for mtnt
    "trg": "ja",
    "model_trg_code": "ja_XX",
    "data_trg_code": "ja_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "mtnt"
}

mtnt_en_fr_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX", # not needed for mtnt
    "trg": "fr",
    "model_trg_code": "fr_XX",
    "data_trg_code": "fr_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "mtnt"
}

iwslt17_en_fr_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX", # not needed for mtnt
    "trg": "fr",
    "model_trg_code": "fr_XX",
    "data_trg_code": "fr_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "iwslt2017"
}


wmt19_en_gu_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "gu",
    "trg_code": "gu_IN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "wmt19"
}

wmt19_en_kk_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "kk",
    "model_trg_code": "kk_KZ",
    "data_trg_code": "kk_KZ",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "wmt19"
}


wmt19_en_kk_mt5_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "kk",
    "trg_code": "kk_KZ",
    "model_name_or_path": "google/mt5-small",
    "dataset": "wmt19"
}


wmt19_en_kk_mbart50_mmt_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "kk",
    "model_trg_code": "kk_KZ",
    "data_trg_code": "kk_KZ",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "wmt19"
}

flores_en_ne_mbart50_config = {
    "src": "eng_Latn",
    "src_code": "en_XX",
    "trg": "npi_Deva",
    "trg_code": "ne_NP",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "facebook/flores"
}
    
flores_en_si_mbart50_config = {
    "src": "eng_Latn",
    "src_code": "en_XX",
    "trg": "sin_Sinh",
    "trg_code": "si_LK",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "facebook/flores"
}


wmt19_en_lt_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "lt",
    "model_trg_code": "lt_LT",
    "data_trg_code": "lt_LT",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "wmt19"
}

wmt19_en_lt_mbart50_mmt_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "lt",
    "model_trg_code": "lt_LT",
    "data_trg_code": "lt_LT",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "wmt19"
}

wmt19_en_fi_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "fi",
    "trg_code": "fi_FI",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "wmt19"
}

wmt19_en_zh_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "zh",
    "trg_code": "zh_CN",
    "model_name_or_path": "facebook/mbart-large-50",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "zh_core_web_trf",
    "dataset": "wmt19"
}

wmt19_en_de_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "de",
    "trg_code": "de_DE",
    "model_name_or_path": "facebook/mbart-large-50",
    "spacy_src": "en_core_web_md",
    "spacy_trg": "de_dep_news_trf",
    "dataset": "wmt19"
}

ML50_en_bn_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "bn",
    "model_trg_code": "bn_IN",
    "data_trg_code": "bn_IN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_bn_en_mbart50_config = {
    "src": "bn",
    "model_src_code": "bn_IN",
    "data_src_code": "bn_IN",
    "trg": "en",
    "model_trg_code": "en_XX",
    "data_trg_code": "en_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_mr_en_mbart50_config = {
    "src": "mr",
    "model_src_code": "mr_IN",
    "data_src_code": "mr_IN",
    "trg": "en",
    "model_trg_code": "en_XX",
    "data_trg_code": "en_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_bn_mbart50_mmt_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "bn",
    "model_trg_code": "bn_IN",
    "data_trg_code": "bn_IN",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "ML50"
}

ML50_en_az_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "az",
    "model_trg_code": "az_AZ",
    "data_trg_code": "az_AZ",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_az_nllb_config = {
    "src": "en",
    "model_src_code": "eng_Latn",
    "data_src_code": "en_XX",
    "trg": "az",
    "model_trg_code": "azb_Arab",
    "data_trg_code": "az_AZ",
    "model_name_or_path": "facebook/nllb-200-distilled-600M",
    "dataset": "ML50"
}

ML50_en_mr_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "mr",
    "model_trg_code": "mr_IN",
    "data_trg_code": "mr_IN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_mr_nllb_config = {
    "src": "en",
    "model_src_code": "eng_Latn",
    "data_src_code": "en_XX",
    "trg": "mr",
    "model_trg_code": "mar_Deva",
    "data_trg_code": "mr_IN",
    "model_name_or_path": "facebook/nllb-200-distilled-600M",
    "dataset": "ML50"
}


ML50_en_gl_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "gl",
    "model_trg_code": "gl_ES",
    "data_trg_code": "gl_ES",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_gu_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "gu",
    "model_trg_code": "gu_IN",
    "data_trg_code": "gu_IN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}


ML50_en_mn_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "mn",
    "model_trg_code": "mn_MN",
    "data_trg_code": "mn_MN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Swedish
ML50_en_sv_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "sv",
    "trg_code": "sv_SE",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Thai
ML50_en_th_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "th",
    "trg_code": "th_TH",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Indonesian
ML50_en_id_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "id",
    "model_trg_code": "id_ID",
    "data_trg_code": "id_ID",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Portuguese
ML50_en_pt_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "pt",
    "model_trg_code": "pt_XX",
    "data_trg_code": "pt_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_pt_mbart50_mmt_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "pt",
    "model_trg_code": "pt_XX",
    "data_trg_code": "pt_XX",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "ML50"
}

# Urdu
ML50_en_ur_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "ur",
    "model_trg_code": "ur_PK",
    "data_trg_code": "ur_PK",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_ur_mbart50_mmt_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "ur",
    "model_trg_code": "ur_PK",
    "data_trg_code": "ur_PK",
    "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
    "dataset": "ML50"
}

ML50_en_ur_xlm_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ur",
    "trg_code": "ur_PK",
    "model_name_or_path": "xlm-mlm-tlm-xnli15-1024",
    "dataset": "ML50"
}

# Macedonian
ML50_en_mk_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "mk",
    "trg_code": "mk_MK",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Telugu
ML50_en_te_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "te",
    "trg_code": "te_IN",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Slovenian
ML50_en_sl_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "sl",
    "trg_code": "sl_SI",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

# Georgian
ML50_en_ka_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "ka",
    "model_trg_code": "ka_GE",
    "data_trg_code": "ka_GE",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_ka_en_mbart50_config = {
    "src": "ka",
    "model_src_code": "ka_GE",
    "data_src_code": "ka_GE",
    "trg": "en",
    "model_trg_code": "en_XX",
    "data_trg_code": "en_XX",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "ML50"
}

ML50_en_ka_nllb_config = {
    "src": "en",
    "model_src_code": "eng_Latn",
    "data_src_code": "en_XX",
    "trg": "ka",
    "model_trg_code": "kat_Geor",
    "data_trg_code": "ka_GE",
    "model_name_or_path": "facebook/nllb-200-distilled-600M",
    "dataset": "ML50"
}

ML50_en_ka_xlm_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ka",
    "trg_code": "ka_GE",
    "model_name_or_path": "xlm-mlm-tlm-xnli15-1024",
    "dataset": "ML50"
}

# Romanian
wmt16_en_ro_mbart50_config = {
    "src": "en",
    "src_code": "en_XX",
    "trg": "ro",
    "trg_code": "ro_RO",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "wmt16"
}

iwslt17_en_ko_nllb_config = {
    "src": "en",
    "model_src_code": "eng_Latn",
    "data_src_code": "eng_Latn",
    "trg": "ko",
    "model_trg_code": "kor_Hang",
    "data_trg_code": "kor_Hang",
    "model_name_or_path": "facebook/nllb-200-distilled-600M",
    "dataset": "iwslt2017"
}

feedbackmt_en_de_mbart50_config = {
    "src": "en",
    "model_src_code": "en_XX",
    "data_src_code": "en_XX",
    "trg": "de",
    "model_trg_code": "de_DE",
    "data_trg_code": "de_DE",
    "model_name_or_path": "facebook/mbart-large-50",
    "dataset": "feedbackmt_highres"
}

configs = {
    "ko_en_config": ko_en_config,
    "de_en_config": de_en_config,

    "opus_en_ko_pl_mbart50_config": opus_en_ko_pl_mbart50_config,
    "opus_en_ko_pl_mbart50_config_fine": opus_en_ko_pl_mbart50_config_fine,
    "opus_en_ko_pl_mbart50_mmt_config": opus_en_ko_pl_mbart50_mmt_config,
    "opus_en_ko_pl_mbartcc25_config": opus_en_ko_pl_mbartcc25_config,

    "iwslt17_en_ko_mbart50_config": iwslt17_en_ko_mbart50_config,
    "iwslt17_en_ko_xlmr_config": iwslt17_en_ko_xlmr_config,
    "iwslt17_en_ja_mbart50_config": iwslt17_en_ja_mbart50_config,
    "iwslt17_en_ko_mt5_config": iwslt17_en_ko_mt5_config,
    "iwslt17_en_ko_bart_config": iwslt17_en_ko_bart_config,
    "iwslt17_en_ko_nllb_config": iwslt17_en_ko_nllb_config,
    "iwslt17_en_de_mbart50_config": iwslt17_en_de_mbart50_config,
    "iwslt17_en_de_mbart50_mmt_config": iwslt17_en_de_mbart50_mmt_config,
    "iwslt17_en_zh_mbart50_config": iwslt17_en_zh_mbart50_config,
    "iwslt17_en_fr_mbart50_config": iwslt17_en_fr_mbart50_config,

    "iwslt17_ko_en_mbart50_config": iwslt17_ko_en_mbart50_config,

    "wmt19_en_gu_mbart50_config": wmt19_en_gu_mbart50_config,
    "wmt19_en_kk_mbart50_config": wmt19_en_kk_mbart50_config,
    "wmt19_en_kk_mbart50_mmt_config": wmt19_en_kk_mbart50_mmt_config,
    "wmt19_en_kk_mt5_config": wmt19_en_kk_mt5_config,

    "flores_en_si_mbart50_config": flores_en_si_mbart50_config,
    "flores_en_ne_mbart50_config": flores_en_ne_mbart50_config,

    "wmt19_en_lt_mbart50_config": wmt19_en_lt_mbart50_config,
    "wmt19_en_lt_mbart50_mmt_config": wmt19_en_lt_mbart50_mmt_config,
    "wmt19_en_fi_mbart50_config": wmt19_en_fi_mbart50_config,

    "wmt19_en_zh_mbart50_config": wmt19_en_zh_mbart50_config,
    "wmt19_en_de_mbart50_config": wmt19_en_de_mbart50_config,
    
    "ML50_en_az_mbart50_config": ML50_en_az_mbart50_config,
    "ML50_en_az_nllb_config": ML50_en_az_nllb_config,
    "ML50_en_gl_mbart50_config": ML50_en_gl_mbart50_config,
    "ML50_en_gu_mbart50_config": ML50_en_gu_mbart50_config,
    "ML50_en_mn_mbart50_config": ML50_en_mn_mbart50_config,
    "ML50_en_mr_mbart50_config": ML50_en_mr_mbart50_config,
    "ML50_mr_en_mbart50_config": ML50_mr_en_mbart50_config,
    "ML50_en_mr_nllb_config": ML50_en_mr_nllb_config,
    "ML50_en_bn_mbart50_config": ML50_en_bn_mbart50_config,
    "ML50_bn_en_mbart50_config": ML50_bn_en_mbart50_config,
    "ML50_en_bn_mbart50_mmt_config": ML50_en_bn_mbart50_mmt_config,
    "ML50_en_sv_mbart50_config": ML50_en_sv_mbart50_config,
    "ML50_en_th_mbart50_config": ML50_en_th_mbart50_config,
    "ML50_en_id_mbart50_config": ML50_en_id_mbart50_config,
    "ML50_en_pt_mbart50_config": ML50_en_pt_mbart50_config,
    "ML50_en_pt_mbart50_mmt_config": ML50_en_pt_mbart50_mmt_config,
    "ML50_en_ur_mbart50_config": ML50_en_ur_mbart50_config,
    "ML50_en_ur_mbart50_mmt_config": ML50_en_ur_mbart50_mmt_config,
    "ML50_en_ur_xlm_config": ML50_en_ur_xlm_config,
    "ML50_en_mk_mbart50_config": ML50_en_mk_mbart50_config,
    "ML50_en_te_mbart50_config": ML50_en_te_mbart50_config,
    "ML50_en_sl_mbart50_config": ML50_en_sl_mbart50_config,
    "ML50_en_ka_mbart50_config": ML50_en_ka_mbart50_config,
    "ML50_ka_en_mbart50_config": ML50_ka_en_mbart50_config,
    "ML50_en_ka_nllb_config": ML50_en_ka_nllb_config,
    "ML50_en_ka_xlm_config": ML50_en_ka_xlm_config, 
    "wmt16_en_ro_mbart50_config": wmt16_en_ro_mbart50_config,

    "mtnt_en_fr_mbart50_config": mtnt_en_fr_mbart50_config,
    "mtnt_en_ja_mbart50_config": mtnt_en_ja_mbart50_config,

    "robust_en_de_mbart50_config": robust_en_de_mbart50_config,
    "feedbackmt_en_de_mbart50_config": feedbackmt_en_de_mbart50_config

}