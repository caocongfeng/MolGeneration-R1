import os
import torch
import json
import re
import numpy as np
from datasets import load_dataset, Dataset

import wandb

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from torch.utils.data import DataLoader
from tqdm import tqdm

RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

def is_SMILES(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else False

def are_smiles_similar(smiles1, smiles2, threshold=1.0):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    fp1 = generator.GetFingerprint(mol1)
    fp2 = generator.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def read_cot_data(cot_path):
    cot_data = []
    with open(cot_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            cot_data.append({"answer": data["smile"], "des": data["explain"], "cot": data["cot"]})
    return cot_data

# def build_messages_from_cot(cot_data):
#     messages_list = []
#     for i in cot_data:
#         messages = [
#             {"role": "system", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
#             {"role": "user", "content": f"You are an expert...best possible answer for the molecular description: {i['des']}"},
#             {"role": "assistant", "content": f"<Thinking>\n{i['cot']}"}
#         ]
#         messages[2]['content'] = messages[2]['content'].replace("<answer>", "<Answer>\n").replace("</answer>", "\n</Answer>")
#         before, sep, after = messages[2]['content'].rpartition("<Answer>")
#         messages[2]['content'] = before + "\n</Thinking>\n" + sep + after if sep else messages[2]['content']
#         messages_list.append({"messages": messages})
#     return messages_list
    
def build_test_messages(test_data):
    messages_list=[]
    for i in test_data:
        messages = [
            {"role": "system", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation. Could you please provide the SMILES representation based on this molecular description: {} \n 1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. 2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description.".format(i["des"])},
        ]
        messages_list.append({"messages": messages})
    return messages_list
def build_messages_from_cot(cot_data):
    messages_list=[]
    for i in cot_data:
        messages = [
            {"role": "system", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation, professional and rigorous."},
            {"role": "user", "content": "You are an expert in info-chemistry, especially in SMILES interpretation and translation. Could you please provide the SMILES representation based on this molecular description: {} \n 1. Analyze it step by step within the <Thinking></Thinking> tags and give only the SMILES within the <Answer></Answer> tags. 2. Make sure your SMILES is completely correct and is the best possible answer for the molecular description.".format(i["des"])},
            {"role": "assistant", "content": "<Thinking>\n{}".format(i["cot"])}
            # Here is the analysis and the result. delete?
        ]

        messages[2]['content']=messages[2]['content'].replace("<answer>", "<Answer>\n").replace("</answer>", "\n</Answer>")
        marker="<Answer>"
        insert = "\n</Thinking>\n"
        before, sep, after = messages[2]['content'].rpartition(marker)
        messages[2]['content']=(before + insert + sep + after if sep else s)
        messages_list.append({"messages": messages})
    return messages_list

def extract_last_tags(text):
    answer_pattern = re.compile(r"<Answer>(.*?)</Answer>", re.DOTALL)
    thinking_pattern = re.compile(r"<Thinking>(.*?)</Thinking>", re.DOTALL)
    answers = list(re.finditer(answer_pattern, text))
    thinkings = list(re.finditer(thinking_pattern, text))
    answer = answers[-1].group(1).strip() if answers else ""
    thinking = thinkings[-1].group(1).strip() if thinkings else ""
    return answer, thinking

def SFT(projectname, wandbname, finetune_name, output_dir, lr, Full_tuning, lora_rank, save_steps, zero_shot, lora_dropout, num_train_epochs, BATCH_SIZE=4, test_path="./RL_data/test_data.txt"):
    
    wandb.login()

    wandb.init(
        project=projectname, 
        entity="byfrfy",
        name=wandbname,
        )

    # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetune_name, trust_remote_code=True, padding_side='left')
    # model = AutoModelForCausalLM.from_pretrained(
    #     pretrained_model_name_or_path=finetune_name,
    #     # attn_implementation="flash_attention_2",
    #     device_map="auto",
    #     trust_remote_code=True,
    #     # torch_dtype=torch.bfloat16,
    #     # force_download=True,
    # )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=finetune_name, trust_remote_code=True, padding_side='left', force_download=True)
    #  accelerate the training process
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=finetune_name, 
        attn_implementation="flash_attention_2", 
        # device_map="auto", 
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)


    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    SFTargs = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",
        learning_rate=lr,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        push_to_hub=False,
        report_to="wandb",
        max_seq_length=1024,
        # bf16=True,
    )

    train_data = Dataset.from_list(build_messages_from_cot(read_cot_data("./RL_data/train_data.jsonl")))
    val_data = Dataset.from_list(build_messages_from_cot(read_cot_data("./RL_data/val_data.jsonl")))

    trainer = SFTTrainer(
        model=model,
        args=SFTargs,
        train_dataset=train_data,
        eval_dataset=val_data,
        # peft_config=None if Full_tuning else peft_config,
        
    )

    trainer.train()
    trainer.save_model()


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
            print("Thinking: ", thinking)
            

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

    # metrics = trainer.evaluate()
    # print(metrics)
    #################################
    ## test data evaluate
    # #################################
    # print("="*10+"Loading test data..."+"="*10)

    # answers, raw_test_data = [], []
    # with open(test_path, 'r') as f:
    #     for line in f:
    #         try:
    #             answer, des = line.strip().split('\t')
    #             answers.append(answer)
    #             raw_test_data.append({"des": des, "answer": answer})
    #         except:
    #             continue

    # messages = Dataset.from_list([{
    #     "messages": [
    #         {"role": "system", "content": "You are an expert..."},
    #         {"role": "user", "content": f"Could you please...{x['des']}"}
    #     ]
    # } for x in raw_test_data])["messages"]

    # dataloader = DataLoader(messages, batch_size=BATCH_SIZE, collate_fn=list)
    # EM, similarity_score, validation_score = [], [], []

    # for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
    #     texts = [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True) for item in batch]
    #     model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
    #     with torch.no_grad():
    #         outputs = model.generate(**model_inputs, max_new_tokens=1024, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    #     responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    #     for j, response in enumerate(responses):
    #         pred, _ = extract_last_tags(response)
    #         ref = answers[i * BATCH_SIZE + j]
    #         if pred:
    #             valid = is_SMILES(pred)
    #             if valid:
    #                 sim = are_smiles_similar(valid, is_SMILES(ref))
    #                 similarity_score.append(sim)
    #                 validation_score.append(1.0)
    #                 EM.append(1.0 if sim == 1.0 else 0.0)
    #             else:
    #                 similarity_score.append(0.0)
    #                 validation_score.append(0.0)
    #                 EM.append(0.0)
    #         else:
    #             similarity_score.append(0.0)
    #             validation_score.append(0.0)
    #             EM.append(0.0)

    # print("Mean EM:", np.mean(EM))
    # print("Mean Similarity:", np.mean(similarity_score))
    # print("Mean Validation:", np.mean(validation_score))

    # wandb.log({"mean EM": np.mean(EM), "mean similarity": np.mean(similarity_score), "mean validation": np.mean(validation_score)})

if __name__ == '__main__':
    import argparse
    def str2bool(v):
        return v.lower() in ('yes', 'true', 't', 'y', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--projectname', type=str, default="Qwen_SFT")
    parser.add_argument("--wandbname", type=str, default="Qwen_SFT_FULL")
    parser.add_argument("--finetune_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output_dir", type=str, default="./SFT_models/Qwen7B_Full_SFT")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--Full_tuning", type=str2bool, default=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--zero_shot", type=str2bool, default=True)
    parser.add_argument("--lora_dropout", type=float, default=0.06)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()

    SFT(
        projectname=args.projectname,
        wandbname=args.wandbname,
        finetune_name=args.finetune_name,
        output_dir=args.output_dir,
        lr=args.lr,
        Full_tuning=args.Full_tuning,
        lora_rank=args.lora_rank,
        save_steps=args.save_steps,
        zero_shot=args.zero_shot,
        lora_dropout=args.lora_dropout,
        num_train_epochs=args.num_train_epochs
    )