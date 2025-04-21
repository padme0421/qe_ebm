import json
import openai
import dotenv
import os
import re

openai.api_key = dotenv.get_key("../.env", 'OPENAI_KEY')

def translate(tgt_lang: str, tokens, model: str, run_id: str, algorithm: str, sample_id: int):

    system_message = f"You are an assistant."
    prompt = f"""Provide English glosses for the following list of {tgt_lang} tokens:
            {tokens}
            Output only a python list, with no comments.
            """
    
    messages = [{"role": "system", "content": system_message},
            {"role": "user", "content": prompt}]

    try:
        response = openai.ChatCompletion.create(model=model, messages=messages,
                                            temperature=0.8).choices[0].message.content
        pattern = r"\[.*?\]"
        response = re.findall(pattern, response)[0]

        gloss_file = f"{run_id}/{algorithm}_token_gloss_id{sample_id}.txt"
        with open(gloss_file, 'w') as f:
            f.write(response)

    except Exception as e:
        print(e)
        print("Error in LLM translation")
        response = "ERROR"
    
    return response

if __name__ == "__main__":
    run_id = "773mpytd"
    algorithm = "reinforce"
    tgt_lang = "Bengali"
    model = "gpt-4o"

    for sample_id in range(216):

        grads_file = f"{run_id}/{algorithm}_grads_id{sample_id}.pkl"
        strings_file = f"{run_id}/{algorithm}_strings_id{sample_id}.json"
        tokens_file = f"{run_id}/{algorithm}_tokens_id{sample_id}.txt"

        if not os.path.isfile(grads_file):
            continue

        with open(strings_file) as f:
            strings_dict = json.load(f)
        with open(tokens_file) as f:
            tokens = json.load(f)
        # keys: input, labels, output

        translate(tgt_lang, tokens, model, run_id, algorithm, sample_id)

