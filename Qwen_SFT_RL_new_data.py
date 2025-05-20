import os
import wandb
import torch
import argparse
from datasets import load_dataset, Dataset

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import rdFingerprintGenerator

import re

from transformers import TextStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm  # 

import json

from trl import GRPOConfig, GRPOTrainer

from peft import LoraConfig

from peft import PeftModel

RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

SYSTEM_PROMPT = """You are an expert in info-chemistry, especially in SMILES generation, professional and rigorous. """


def read_data(path: str):
    molecle_answer_cot_list=[]
    with open(path,'r') as f:
        for i in f.readlines():
            socre, molecule, answer, molecle_des, cot=i.split('\t')
            # print(i)
            molecle_answer_cot_list.append({"answer":answer, "des":molecle_des,"cot":cot})
    return molecle_answer_cot_list

def read_cot_data(cot_path):
    cot_data = []
    with open(cot_path, 'r') as f:
        for line in f:
            data= json.loads(line)
            cot_data.append({"answer":data["smile"],"des":data["explain"],"cot":data["cot"]})
    return cot_data

def read_chebi_data(path: str):
    molecle_answer_cot_list=[]
    with open(path,'r') as f:
        for i in f.readlines():
            answer, molecle_des =i.split('\t')
            # print(i)
            molecle_answer_cot_list.append({"answer":answer, "des":molecle_des})
    return molecle_answer_cot_list
    

def build_QA_dict(molecule_data, zero_shot=True):
    if zero_shot:
        return [{"question":"You are an expert in info-chemistry, especially in SMILES interpretation and translation. Could you please provide the SMILES representation based on this molecular description: {} 1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. 2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description.".format(i["des"]), 'answer':i["answer"]} for i in molecule_data]
    else:
        return [{"question":"You are an expert in info-chemistry, especially in SMILES interpretation and translation. 1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. 2. Double-check it within the <Thinking></Thinking> tags and ensure that the SMILES you generate is completely correct. 3. Make sure your answer is the best possible answer for the molecular description. Here is an example: \n<Thinking>\nTo derive the SMILES representation for nonadecanoic acid, let\'s break down its structure: 1. **Base Structure**: Nondadecanoic Acid Structure (also known as Mollardic Acid):   - Nonadecanoic acid has a carbon chain of 9 carbons (C9) connected to a carboxylic acid group (-COOH). 2. **Double Bond Position**: Since the double bond is specified to occur between C10 and C11, we can represent this as `=`.3. **Identifying Hydrogens**:    - A terminal acyl (carbon) from the carboxylic acid end contributes one hydrogen atom.   - Each carbon in the chain except the carboxylic carbon (which may have fewer hydrogens), typically connects to 2 or 3 hydrogens but since a terminal carbon in this case also connects to 2 additional carbons, contributing 4 hydrogens would be reasonable.4. **Constructing the SMILES**:   - Start with the carboxylic acid part, contribute \"C(=O)O\" which denotes the carboxylic acid functionality, converting it to just \"C\".   - The rest of the carbon chain remains intact, connecting as needed.   Wait! Let's check all the information we have to ensure that the SMILES we generate is completely correct. Yes, looks good! The final SMILES representation for nonadecanoic acid described above is: \n</Thinking>\n<Answer>\nCCCCCCC/C=C/CCCCCC(=O)O\n</Answer>\n Could you please provide the SMILES representation based on this molecular description: {} ".format(i["des"]), 'answer':i["answer"]} for i in molecule_data]

def get_molecule_questions(QA_dict):
    QA_dict = QA_dict.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['answer']
    }) # type: ignore
    return QA_dict # type: ignore

    ##### soft format conditional reward+samilarity score

def extract_answer(text: str) -> str:

    answer_pattern   = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL)
    answers   = list(re.finditer(answer_pattern, text))
    answer   = answers[-1].group(1).strip() if answers else ""

    return  answer

def extract_last_tags(text: str):
    answer_pattern   = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL)
    thinking_pattern = re.compile(r"<Thinking>(.*?)</Thinking>", re.DOTALL)

    answers   = list(re.finditer(answer_pattern, text))
    thinkings = list(re.finditer(thinking_pattern, text))

    answer   = answers[-1].group(1).strip() if answers else ""
    thinking = thinkings[-1].group(1).strip() if thinkings else ""

    return  answer,thinking

def is_SMILES(smiles):
    """Check if a given string is a valid SMILES representation.
    
    Returns:
    - Canonical SMILES string if valid.
    - False if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else False

def are_smiles_similar(smiles1, smiles2, threshold=1.0):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0.0  # Invalid SMILES
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity
##### reward
# ! only satisfy the soft format can do the is smiles reward
## [0.0-1.0]
def is_SMILES_reward(completions, **kwargs):
    # print(completions)
    """Reward function that rewards if both <thinking> and <answer> blocks exist."""
    # pattern = re.compile(r"<Thinking>.*?</Thinking>\s*<Answer>.*?</Answer>", re.DOTALL)
    # responses = [completion[0]["content"] for completion in completions]
    # matches = [bool(pattern.search(r)) for r in responses]
    # print("matches: ",matches)

    # if matches[0] == True:
    responses = [completion[0]['content'] for completion in completions]
    extract_smiles = [extract_answer(r) for r in responses]

    print("Extract Smiles",extract_smiles)
    print("is_SMILES_reward",[1.0 if is_SMILES(s) else 0.0 for s in extract_smiles])
    return [1.0 if is_SMILES(s) else 0.0 for s in extract_smiles]
    # else:
    #     responses = [completion[0]['content'] for completion in completions]
    #     extract_smiles = [extract_answer(r) for r in responses]

    #     print("Extract Smiles",extract_smiles)
    #     print("is_SMILES_reward",[0.0 for s in responses])
    #     return [0.0 for s in responses]
        

# ! using are_smiles_similar get the similarity 
# ! [0.0-4.0]
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_answer(r) for r in responses]
#     print('='*30+'\n')
#     print('Responses: ',responses)
#     print('Answer: ',answer)
#     print('Extracted Answers: ',extracted_responses)
#     # print('-'*20+'\n', f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
#     smiless=[is_SMILES(r) for r in extracted_responses]
#     # print('Is SMILES: ',smiless)
#     answers=[is_SMILES(a) for a in answer]
#     # print('Answers: ',answers)
#     print("Correctness Reward", [0.0 if s is False else 4.0*are_smiles_similar(s,a) for s,a in zip(smiless,answers)])
#     # False
#     return [0.0 if s is False else 4.0*are_smiles_similar(s,a) for s,a in zip(smiless,answers)]
# ! using are_smiles_similar get the similarity 
# ! [0.0-4.0]
# ! using the EM to get the reward
# ! using are_smiles_similar get the similarity 
# ! [0.0-4.0]
# ! using the EM to get the reward
def are_smiles_high_similar(smiles1, smiles2, threshold=0.5):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0.0  # Invalid SMILES
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return 1.0 if similarity>= threshold else 0.0

def are_smiles_EM(smiles1, smiles2, threshold=1.0):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return 0.0  # Invalid SMILES
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    # 1.0 if similarity==1.0 else 0.0
    return 1.0 if similarity==1.0 else 0.0

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_answer(r) for r in responses]
    print('='*30+'\n')
    print('Responses: ',responses)
    print('Answer: ',answer)
    print('Extracted Answers: ',extracted_responses)
    # print('-'*20+'\n', f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    smiless=[is_SMILES(r) for r in extracted_responses]
    # print('Is SMILES: ',smiless)
    answers=[is_SMILES(a) for a in answer]
    # print('Answers: ',answers)
    print("Correctness Reward", [0.0 if s is False else 4.0*are_smiles_EM(s,a) for s,a in zip(smiless,answers)])
    # False
    return [0.0 if s is False else 4.0*are_smiles_EM(s,a) for s,a in zip(smiless,answers)]

# def are_smiles_EM(smiles1, smiles2, threshold=1.0):
#     mol1 = Chem.MolFromSmiles(smiles1)
#     mol2 = Chem.MolFromSmiles(smiles2)

#     if mol1 is None or mol2 is None:
#         return 0.0  # Invalid SMILES
#     generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    
#     fp1 = generator.GetFingerprint(mol1)
#     fp2 = generator.GetFingerprint(mol2)

#     similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
#     # 1.0 if similarity==1.0 else 0.0
#     return 1.0 if similarity==1.0 else 0.0

# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_answer(r) for r in responses]
#     print('='*30+'\n')
#     print('Responses: ',responses)
#     print('Answer: ',answer)
#     print('Extracted Answers: ',extracted_responses)
#     # print('-'*20+'\n', f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
#     smiless=[is_SMILES(r) for r in extracted_responses]
#     # print('Is SMILES: ',smiless)
#     answers=[is_SMILES(a) for a in answer]
#     # print('Answers: ',answers)
#     print("Correctness Reward", [0.0 if s is False else 4.0*are_smiles_EM(s,a) for s,a in zip(smiless,answers)])
#     # False
#     return [0.0 if s is False else 4.0*are_smiles_EM(s,a) for s,a in zip(smiless,answers)]

# ! [0.0-1.0]
  # ideal length
def len_reward(completions, max_len = 1000, **kwargs):
    responses = [completion[0]["content"] for completion in completions]

    print("Len",[len(response) for response in responses])
    print("Len Reward",[(len(response) / max_len) if len(response) <= max_len else max(0, 1 - (len(response) - max_len) / max_len) for response in responses])
    return [(len(response) / max_len) if len(response) <= max_len else max(0, 1 - (len(response) - max_len) / max_len) for response in responses]
# ! strict it in the format
# ! [0.0-1.0]
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<Thinking>\n.*?\n</Thinking>\n<Answer>\n.*?\n</Answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    print("Strict Format Reward",[1.0 if match else 0.0 for match in matches])
    return [1.0 if match else 0.0 for match in matches]

# ! soft it in the <Thinking>.*?</Thinking> format
# ! [0.0-.1]
def soft_thinking_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that rewards if <thinking>...</thinking> exists anywhere."""
    pattern = re.compile(r"<Thinking>.*?</Thinking>", re.DOTALL)
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(pattern.search(r)) for r in responses]
    print("Soft Thinking Reward",[0.1 if match else 0.0 for match in matches])
    return [0.1 if match else 0.0 for match in matches]

# ! soft it in the <Answer>.*?</Answer> format
# ! [0.0-.1]
def soft_answer_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that rewards if <answer>...</answer> exists anywhere."""
    pattern = re.compile(r"<Answer>.*?</Answer>", re.DOTALL)
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(pattern.search(r)) for r in responses]
    print("Soft Answer Reward",[0.1 if match else 0.0 for match in matches])
    return [0.1 if match else 0.0 for match in matches]

# ! soft it in the <Thinking>.*?</Thinking><Answer>.*?</Answer> format
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that rewards if both <thinking> and <answer> blocks exist."""
    pattern = re.compile(r"<Thinking>.*?</Thinking>\s*<Answer>.*?</Answer>", re.DOTALL)
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(pattern.search(r)) for r in responses]
    print("Soft Format Reward", [0.2 if match else 0.0 for match in matches])
    return [0.2 if match else 0.0 for match in matches]


# line stack to keep correctness
def stack_reward(completions, answer, **kwargs):
    # print(completions)
    """Reward function that rewards if both <thinking> and <answer> blocks exist."""
    max_len=1350
    pattern = re.compile(r"<Thinking>.*?</Thinking>\s*<Answer>.*?</Answer>", re.DOTALL)
    ## fit soft format then SMILES:
    reward=[]
    extract_smiles=[]
    matches=[]
    smiless=[]

    responses = [completion[0]["content"] for completion in completions]
    print("responses: ",responses)

    for (i,completion) in enumerate(completions):

        response=completion[0]['content']
        extract_smile=extract_answer(response)
        extract_smiles.append(extract_smile)
        match=bool(pattern.search(response))
        matches.append(match)

        ## format match
        if match !=False:
            ## is_SMILES_reward
            ismiles=is_SMILES(extract_smile)
            smiless.append(ismiles)
            if ismiles!=False and extract_smile!='':

                similariry_score=are_smiles_EM(ismiles,is_SMILES(answer[i]))
                # reward.append(2.0*similariry_score+1.0)
                if len(response) <= max_len:
                    lenth_reward=len(response) / max_len
                    if similariry_score ==1.0:
                        reward.append(2.0*similariry_score+1.0+lenth_reward)
                    else:
                        reward.append(0.0)
                else:
                    lenth_reward=max(0, 1 - (len(response) - max_len) / max_len)
                    if similariry_score ==1.0:
                        reward.append(2.0*similariry_score+1.0+lenth_reward)
                    else:
                        reward.append(0.0)

            else:
                reward.append(0.0)


        else:
            reward.append(0.0)
            smiless.append(False)
    print("Answer", answer)
    print("pattern matches: ", matches)
    print("Extract Smiles", extract_smiles)
    print("Is smiless", smiless)
    print("Correctness reward", reward)

    return reward

def build_test_messages(test_data):
    messages_list=[]
    for i in test_data:
        messages = [
            {"role": "system", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation. Could you please provide the SMILES representation based on this molecular description: {} \n 1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. 2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description.".format(i["des"])},
        ]
        messages_list.append({"messages": messages})
    return messages_list


def Qwen_RL(
    projectname="Qwen_base_full_RL",
    wandbname="Qwen_RL_non_conditional_non_KL",
    tags=["Qwen", "RL", "full"],
    finetune_name = "Qwen/Qwen2.5-7B",
    finetune_tags = ["molecule", "full tuning", "RL", "Qwen"],
    output_dir = "Qwen_base_full_RL_KL_softformat_conditional_similarity",
    conditional_reward_style=True,
    KL=0.0, #0.2
    temperature=0.7,
    lr=5e-6,
    per_device_train_batch_size = 4,
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 1024,
    num_train_epochs = 3, # Set to 1 for a full training run
    save_steps = 100,
    zero_shot=True,
    BATCH_SIZE=4,
    test_path = "./RL_data/test_data.txt",
    ):

    wandb.login()

    wandb.init(
        project=projectname, 
        entity="byfrfy",
        name=wandbname,
        tags=tags,
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    # print(device)
    # #### ChEBI-20 data
    # chebi_path ="./COT/ChEBI-20"
    # # similarity_rank_path="./COT/similarity_rank_data"
    # test_path=os.path.join(chebi_path,"test.txt")
    # train_path=os.path.join(chebi_path,"train.txt")
    # validation_path=os.path.join(chebi_path,"validation.txt")

    # test=read_chebi_data(test_path)
    # train=read_chebi_data(train_path)
    # validation=read_chebi_data(validation_path)
    # len(test),len(train),len(validation)
    
    # similarity_rank_path="./COT/similarity_rank_data"
    # test_path=os.path.join(similarity_rank_path,"test/1.0_similarity.txt")
    # train_path=os.path.join(similarity_rank_path,"train/1.0_similarity.txt")
    # validation_path=os.path.join(similarity_rank_path,"validation/1.0_similarity.txt")

    # test=read_data(test_path)
    # train=read_data(train_path)
    # validation=read_data(validation_path)
    # len(test),len(train),len(validation)
    # #### COT data
    # cot_path="./COT/results_with_scores.jsonl"

    # cot_data = []
    # with open(cot_path, 'r') as f:
    #     for line in f:
    #         data= json.loads(line)
    #         cot_data.append({"answer":data["smile"],"des":data["explain"],"cot":data["cot"]})
    #         # cot_data.append(json.loads(line))
    # #### Merge cot and trian
    # # train[2],cot_data[2],len(cot_data)
    # train_data = train+cot_data
    # len(train_data),len(train),len(cot_data)
    #### build QA pairs

    train_cot_path="./RL_data/train_data.jsonl"
    val_cot_path="./RL_data/val_data.jsonl"
    test_cot_path="./RL_data/test_data.jsonl"

    train_data = read_cot_data(train_cot_path)
    validation_data = read_cot_data(val_cot_path)
    test = read_cot_data(test_cot_path)

    print("="*20)
    print("train_data: ",len(train_data))
    print("validation_data: ",len(validation_data))
    print("test: ",len(test))

    test_data=build_QA_dict(test, zero_shot=zero_shot)
    train_data=build_QA_dict(train_data, zero_shot=zero_shot)
    validation_data=build_QA_dict(validation_data, zero_shot=zero_shot)
    len(test_data),len(train_data),len(validation_data)


    #! some paper using the system prompt to constrain the format, but it does not work in this case 


    test_data=get_molecule_questions(Dataset.from_list(test_data))
    train_data=get_molecule_questions(Dataset.from_list(train_data))
    validation_data=get_molecule_questions(Dataset.from_list(validation_data))
    #### Load Model
    # finetune_name = finetune_name
    # finetune_tags = finetune_tags

    os.environ["TRANSFORMERS_CACHE"] = "./"
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    #! to use multiple GPUs
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetune_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=finetune_name,device_map="auto")
    model = PeftModel.from_pretrained(model, finetune_name, is_trainable=True)
    #### define rewards
    #! set uo rdkit

    #### TRL GRPOTrainer
    #### `no KL`
    # output_dir = output_dir


    training_args = GRPOConfig(
        # use_vllm = True, # use vLLM for fast inference!
        learning_rate = lr,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_8bit",
        # optim = "adamw_torch",
        #! 
        logging_steps = 1,
        bf16 = True,
        fp16 = False,
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = 2, # Increase to 4 for smoother training
        num_generations = num_generations, # Decrease if out of memory
        max_prompt_length = max_prompt_length,
        max_completion_length = max_completion_length,
        num_train_epochs = num_train_epochs, # Set to 1 for a full training run
        # max_steps = 10,
        # save_steps = save_steps,
        save_strategy = "epoch",
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = output_dir,

        repetition_penalty = 1.2,  # 避免过多重复

        log_completions = True,
        temperature = temperature,
        beta=KL,
        epsilon = 1.2,
        epsilon_high = 2.8,
        loss_type="dr_grpo",  #bnpo #dr_grpo
        mask_truncated_completions = True,

    )
    if conditional_reward_style:
        print("="*10+"conditional_reward_style"+"="*10)
        reward_funcs=[
            strict_format_reward_func,
            # soft_format_reward_func,
            stack_reward,
        ]

    else:
        print("="*10+"no_conditional_reward_style"+"="*10)
        reward_funcs=[
            correctness_reward_func,
            len_reward,
            strict_format_reward_func,
            # soft_thinking_format_reward_func,
            # soft_answer_format_reward_func,
            soft_format_reward_func,
            is_SMILES_reward,
        ]

    # peft_config = LoraConfig.from_pretrained(finetune_name)
    # print(peft_config)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_data,
        eval_dataset = validation_data,

        # peft_config=peft_config,
        # callbacks=[RewardLoggingCallback()],
    )
    # trainer.train(resume_from_checkpoint = True)
    trainer.train()


    #################################
    ## test data evaluate
    #################################

    if "Qwen2.5" in finetune_name or "qwen2.5" in finetune_name:
        model_type = "Qwen2.5"
    elif "Qwen3" in finetune_name or "qwen3" in finetune_name:
        model_type = "Qwen3"
    else:
        model_type = None


    raw_test_data = []
    answers = []
    with open(test_path, 'r') as f:
        for line in f:
            try:
                answer, des = line.strip().split('\t')
                raw_test_data.append({"des": des, "answer": answer})
                answers.append(answer)
            except ValueError:
                continue  # 忽略不规范行

    # 构造消息
    test_dataset = build_test_messages(raw_test_data)
    test_dataset = Dataset.from_list(test_dataset)
    messages = test_dataset["messages"]

    # 设置 batch size（根据显存情况调整）
    dataloader = DataLoader(messages, batch_size=BATCH_SIZE, collate_fn=list)

    # 评估指标列表
    EM = []
    similarity_score = []
    validation_score = []

    # 批量推理
    for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        print(batch)
        if model_type == "Qwen3":
            texts = [
                tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for item in batch
            ]        
        else:
            texts = [
                tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
                for item in batch
            ]
        model_inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # greedy
            )

        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for j, response in enumerate(responses):
            global_idx = i * BATCH_SIZE + j
            ref_answer = answers[global_idx]
            pred_answer, thinking = extract_last_tags(response)

            # Optional debug print
            # if global_idx < 20 or global_idx % 10 == 0:
            print("-"*10 + f"Output Example {global_idx}" + "-"*10)
            print("Reference Answer: ", ref_answer)
            print("Predicted Answer: ", pred_answer)
            print("Response: ", response)
            print("Response Length: ",len(response))
            print("Thinking: ", thinking)
            print("="*20)

            # 计算指标
            if pred_answer:
                if is_SMILES(pred_answer):
                    sim_score = are_smiles_similar(is_SMILES(pred_answer), is_SMILES(ref_answer))
                    similarity_score.append(sim_score)
                    validation_score.append(1.0)
                    EM.append(1.0 if sim_score == 1.0 else 0.0)
                else:
                    similarity_score.append(0.0)
                    validation_score.append(0.0)
                    EM.append(0.0)
            else:
                similarity_score.append(0.0)
                validation_score.append(0.0)
                EM.append(0.0)

            
            print("EM: ", EM[-1])
            print("Similarity Score: ", similarity_score[-1])
            print("Validation Score: ", validation_score[-1])
            print("-"*10 + "End Example" + "-"*10)


    # 打印结果
    print("="*10 + "Final Evaluation Results" + "="*10)
    print("Mean EM: ", np.mean(EM))
    print("Mean Similarity: ", np.mean(similarity_score))
    print("Mean Validation Rate: ", np.mean(validation_score))
    print("_"*40)


    wandb.log({
        "mean EM": np.mean(EM),
        "mean similarity": np.mean(similarity_score),
        "mean validation": np.mean(validation_score),
    })
    
if __name__ == '__main__':

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # parser.add_argument("--conditional_reward_style", 
    #                     type=str2bool, 
    #                     default=True)

    parser = argparse.ArgumentParser(description='Qwen_RL')

    parser.add_argument('--projectname', 
                    type=str, default="Qwen_base_full_RL", 
                    help='wandb project name')
    parser.add_argument("--wandbname",
                    type=str, default="Qwen_RL_non_conditional_non_KL", 
                    help="")
    parser.add_argument("--finetune_name",
                    type=str, default="Qwen/Qwen2.5-7B",
                    help="fine tuning models name default Qwen2.5")
    parser.add_argument("--output_dir",
                    type=str, default="Qwen_base_full_RL_KL_softformat_conditional_similarity",
                    help="model save path")

    parser.add_argument("--conditional_reward_style", 
                        type=str2bool, default=True)

    parser.add_argument("--KL",
                        type=float, default=0.0)
    parser.add_argument("--temperature",
                        type=float, default=0.7)
    parser.add_argument("--lr",
                        type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=4)
    parser.add_argument("--num_generations",
                        type=int, default=2)
    parser.add_argument("--max_prompt_length", 
                        type=int, default=1024)
    parser.add_argument("--max_completion_length", 
                        type=int, default=1024)
    parser.add_argument("--num_train_epochs", 
                        type=int, default=3)
    parser.add_argument("--save_steps", 
                        type=int, default=100)
    parser.add_argument("--zero_shot", 
                        type=str2bool, default=True)


    args = parser.parse_args()

    print(args)

    Qwen_RL(
        projectname = args.projectname,
        wandbname = args.wandbname,
        finetune_name = args.finetune_name,
        output_dir = args.output_dir,
        conditional_reward_style = args.conditional_reward_style,
        KL = args.KL,
        temperature = args.temperature,
        lr = args.lr,
        per_device_train_batch_size = args.per_device_train_batch_size,
        num_generations = args.num_generations, # Decrease if out of memory
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs, # Set to 1 for a full training run
        save_steps = args.save_steps,
        zero_shot= args.zero_shot, # zero_shot=True,
        )
