from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from torch.utils.data import DataLoader, Dataset
import json

from torch.nn.utils.rnn import pad_sequence
import torch

from tqdm import tqdm
import os
import copy

from eval_fn import evaluation_korean_contest_culture_QA
from post_processing_with_huristic import MODEL_NAME_to_POST_FUNCTION, MODEL_SPEAKER_TOKENS

class RL_model(torch.nn.Module):
    def __init__(self, model, tokenizer, generate_args, args):
        super(RL_model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.generate_args = generate_args
        self.logging_list = []

        for key in MODEL_NAME_to_POST_FUNCTION.keys():
            if key.lower() in self.args.model_name.lower():
                self.post_process_fn = MODEL_NAME_to_POST_FUNCTION[key]
                print(f"Using post processing function for {key}: {self.post_process_fn}")
                break
            
    def reward_function(self, true, pred):
        scores = []
        
        competition_score = evaluation_korean_contest_culture_QA(true, pred)
        
        if true[0]["input"]["question_type"] == "선다형":
            scores.append(competition_score["accuracy"])
        elif true[0]["input"]["question_type"] == "단답형":
            scores.append(competition_score["exact_match"])
        elif true[0]["input"]["question_type"] == "서술형":
            scores.append((competition_score["rouge_1"]+competition_score["bertscore"]) * self.args.generation_reward_scale)
        
        if self.args.use_format_reward:
            if true[0]["input"]["question_type"] == "서술형":
                if "답변:" in pred[0]["output"]["answer"]:
                    pred[0]["output"]["answer"] = pred[0]["output"]["answer"].split("답변:")[-1].strip()
                if len(pred[0]["output"]["answer"]) > 200 and len(pred[0]["output"]["answer"]) < 600:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            else:
                if "답변:" not in pred[0]["output"]["whole_answer"]:
                    scores.append(0.0)
                elif "생각 과정:" in pred[0]["output"]["whole_answer"].split("답변:")[1]:
                    scores.append(0.0)
                elif "생각 과정:" not in  pred[0]["output"]["whole_answer"].split("답변:")[0]:
                    scores.append(0.0)
                else:
                    if len(pred[0]["output"]["whole_answer"].split("답변:")[0].strip()) < 10:
                        scores.append(0.0)
                    elif len(pred[0]["output"]["whole_answer"].split("생각 과정:")[0].strip()) > 3:
                        scores.append(0.0)
                    else:
                        scores.append(1.0)
        
        print(f"Scores: {scores}")            
        return (sum(scores) / len(scores)) * self.args.reward_scale
        

    def forward(self, input_ids, attention_mask=None, data=None):
        input_b, input_l = input_ids.shape
        
        if input_b > 1:
            raise ValueError("RL model only supports batch size of 1 for now.")
        
        with torch.no_grad():
            self.model.eval()
            self.model.gradient_checkpointing_disable()
            candidate_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generate_args
            )
            
            print(candidate_outputs)
            _, candidate_max_len = candidate_outputs.shape
            answer_max_len = candidate_max_len - input_l
            print(candidate_outputs.shape)
        
            candidate_decoded = [self.tokenizer.decode(output[input_l:], skip_special_tokens=True) for output in candidate_outputs]
            for i, d in enumerate(candidate_decoded):
                print(f"Candidate{i}: {d}\n\n")
            
            candidate_scores = []
            for_answer_data = copy.deepcopy(data)
            for i in range(self.args.num_candidates):
                for_answer_data[0]["output"]["answer"] = candidate_decoded[i].split("답변:")[-1].strip() if "답변:" in candidate_decoded[i] else candidate_decoded[i].strip()
                for_answer_data[0]["output"]["whole_answer"] = candidate_decoded[i].strip()
                for_answer_data[0] = self.post_process_fn(for_answer_data[0])
                
                score = self.reward_function(data, for_answer_data)
                candidate_scores.append(score)
            
            if self.args.use_write_type_answer:
                if data[0]["input"]["question_type"] == "서술형":
                    for_answer_data[0]["output"]["answer"] = data[0]["output"]["answer"]
                    for_answer_data[0]["output"]["whole_answer"] = data[0]["output"]["answer"]
                    for_answer_data[0] = self.post_process_fn(for_answer_data[0])
                    wta_score = self.reward_function(data, for_answer_data)

                    if self.args.wta_reward_stretegy == "cand_include":
                        candidate_scores.append(wta_score)
            
            mean_candiate_scores = sum(candidate_scores) / len(candidate_scores)
            
            candidate_datas = []
            for i in range(self.args.num_candidates):
                input_ids_i = candidate_outputs[i].unsqueeze(0)
                labels_i = input_ids_i.clone()
                labels_i[:, :input_l] = -100  # Mask input tokens
                labels_i[labels_i == self.tokenizer.pad_token_id] = -100  # Mask padding tokens
                labels_i[labels_i == self.tokenizer.eos_token_id] = -100  # Mask eos tokens
                masks = labels_i.not_equal(-100)
                
                adventage = candidate_scores[i] - mean_candiate_scores
                
                candidate_datas.append({
                    "input_ids": input_ids_i,
                    "labels": labels_i,
                    "masks": masks,
                    "reward": candidate_scores[i],
                    "adventage": adventage,
                })
            
            if self.args.use_write_type_answer:
                if data[0]["input"]["question_type"] == "서술형":
                    answer_tokens = self.tokenizer("답변: "+data[0]["output"]["answer"], padding="max_length", max_length=answer_max_len, return_tensors="pt").input_ids.to(input_ids.device)
                    
                    if answer_tokens.shape[1] < answer_max_len:
                        answer_tokens = torch.cat([answer_tokens, torch.tensor([[self.tokenizer.eos_token_id]], device=input_ids.device).repeat(1, answer_max_len - answer_tokens.shape[1])], dim=1)
                    
                    input_ids_i = torch.cat([input_ids.clone(), answer_tokens], dim=1)
                    labels_i = input_ids_i.clone()
                    
                    labels_i[:, :input_l] = -100  # Mask input tokens
                    labels_i[labels_i == self.tokenizer.pad_token_id] = -100  # Mask padding tokens
                    labels_i[labels_i == self.tokenizer.eos_token_id] = -100  # Mask eos tokens
                    masks = labels_i.not_equal(-100)
                    
                    if self.args.wta_reward_stretegy == "cand_max":
                        adventage = max(candidate_scores) - mean_candiate_scores
                    elif self.args.wta_reward_stretegy == "cand_diff" or self.args.wta_reward_stretegy == "cand_include":
                        adventage = wta_score - mean_candiate_scores
                    elif self.args.wta_reward_stretegy == "1":
                        adventage = 1.0
                    
                    candidate_datas.append({
                        "input_ids": input_ids_i,
                        "labels": labels_i,
                        "masks": masks,
                        "reward": wta_score,
                        "adventage": adventage,
                    })
        
        losses = []
        self.model.train()
        self.model.gradient_checkpointing_enable()
        
        for i in range(len(candidate_datas)):
            if candidate_datas[i]["adventage"] == 0:
                loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
                continue
            
            print(f"Candidate {i+1} reward: {candidate_datas[i]['reward']}, advantage: {candidate_datas[i]['adventage']}")
            input_ids_candidate = candidate_datas[i]["input_ids"]
            labels_candidate = candidate_datas[i]["labels"]
            masks_candidate = candidate_datas[i]["masks"]
            
            
            if self.args.rl_mode == "ppo":
                logits = self.model(input_ids=input_ids_candidate).logits
                
                answer_logits = logits[:, input_l - 1:, :].contiguous()
                answer_tokens = input_ids_candidate[:, input_l - 1:].contiguous()
                answer_masks = masks_candidate[:, input_l - 1:].contiguous()
                
                shifted_logits = answer_logits[:, :-1, :].contiguous()
                shifted_labels = answer_tokens[:, 1:].contiguous()
                shifted_masks = answer_masks[:, 1:].contiguous()
                
                candidate_token_probs = torch.nn.functional.softmax(shifted_logits, dim=-1)
                candidate_token_probs = torch.gather(candidate_token_probs, -1, shifted_labels.unsqueeze(-1)).squeeze(-1)
                # candidate_token_probs = candidate_token_probs * shifted_masks.float()
                
                policy_ratio = torch.exp(torch.log(candidate_token_probs + 1e-10) - torch.log(candidate_token_probs.detach() + 1e-10))
                # cliped_policy_ratio = torch.clamp(policy_ratio, 1 - self.args.ppo_clip_range, 1 + self.args.ppo_clip_range) # not use
                loss = -torch.mean(policy_ratio * candidate_datas[i]["adventage"] * shifted_masks.float())
                
                loss = loss / len(candidate_datas) # accumulate loss over candidates
                
                if i < len(candidate_datas) - 1: #24GB에서는 backward여러번해서 batch없이하는게 한계
                    loss.backward()
                
                losses.append(loss)
        
        mean_loss = sum(losses) / len(losses) if losses else torch.tensor(0.0, device=input_ids.device)
        print(f"Total loss: {mean_loss.item()}")
        print(f"Mean candidate score: {mean_candiate_scores}, question type: {data[0]['input']['question_type']}")
        self.logging_list.append({
            "id": data[0]["id"],
            "question_type": data[0]["input"]["question_type"],
            "question": data[0]["input"]["question"],
            "answer": data[0]["output"]["answer"],
            "losses": [l.item() for l in losses],
            "rewards": [d["reward"] for d in candidate_datas],
            "advantages": [d["adventage"] for d in candidate_datas],
            "candidate_decoded": candidate_decoded,
        })
        
        return {"loss": loss}

class CustomCallback(TrainerCallback):
    def __init__(self, args, model, tokenizer, test_fn, gen_args, post_process_fn, dev_dataset=None, test_dataset=None, RL_model=None):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.test_fn = test_fn
        self.gen_args = gen_args
        self.post_process_fn = post_process_fn
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.RL_model = RL_model
        
        os.makedirs(f"results/{args.run_name}/checkpoints", exist_ok=True)
        
    def on_epoch_end(self, args, state, control, **kwargs):
        finished_epochs = int(state.epoch) + 1
        
        if finished_epochs % self.args.testing_every_epoch == 0:
            print(f"Testing after {finished_epochs} epochs...")
            self.gen_args["num_beams"] = 1
            
            try:
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_logging.json", "w") as f:
                    json.dump(self.RL_model.logging_list, f, ensure_ascii=False, indent=4)
                    self.RL_model.logging_list = []  # Clear logging list after saving
            except:
                print(f"Error saving logging file")

            if self.dev_dataset:
                dev_result, dev_whole_data = self.test_fn(
                    args=self.args,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    test_dataset=self.dev_dataset,
                    generate_args=self.gen_args,
                    post_process_fn=self.post_process_fn
                )
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_dev_results.json", "w") as f:
                    json.dump(dev_result, f, ensure_ascii=False, indent=4)
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_dev_whole_data.json", "w") as f:
                    json.dump(dev_whole_data, f, ensure_ascii=False, indent=4)
                
                dev_score = evaluation_korean_contest_culture_QA(self.dev_dataset.data, dev_result)
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_dev_score.json", "w") as f:
                    json.dump(dev_score, f, ensure_ascii=False, indent=4)
            
            if self.test_dataset:
                test_result, test_whole_data = self.test_fn(
                    args=self.args,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    test_dataset=self.test_dataset,
                    generate_args=self.gen_args,
                    post_process_fn=self.post_process_fn
                )
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_test_results.json", "w") as f:
                    json.dump(test_result, f, ensure_ascii=False, indent=4)
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_test_whole_data.json", "w") as f:
                    json.dump(test_whole_data, f, ensure_ascii=False, indent=4)
            
            self.gen_args["num_beams"] = 5
            
            if self.dev_dataset:
                dev_result, dev_whole_data = self.test_fn(
                    args=self.args,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    test_dataset=self.dev_dataset,
                    generate_args=self.gen_args,
                    post_process_fn=self.post_process_fn
                )
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_dev_results_beam.json", "w") as f:
                    json.dump(dev_result, f, ensure_ascii=False, indent=4)
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_dev_whole_data_beam.json", "w") as f:
                    json.dump(dev_whole_data, f, ensure_ascii=False, indent=4)
                
                dev_score = evaluation_korean_contest_culture_QA(self.dev_dataset.data, dev_result)
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_dev_score_beam.json", "w") as f:
                    json.dump(dev_score, f, ensure_ascii=False, indent=4)
            
            if self.test_dataset:
                test_result, test_whole_data = self.test_fn(
                    args=self.args,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    test_dataset=self.test_dataset,
                    generate_args=self.gen_args,
                    post_process_fn=self.post_process_fn
                )
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_test_results_beam.json", "w") as f:
                    json.dump(test_result, f, ensure_ascii=False, indent=4)
                with open(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}_test_whole_data_beam.json", "w") as f:
                    json.dump(test_whole_data, f, ensure_ascii=False, indent=4)
            
            self.model.save_pretrained(f"results/{self.args.run_name}/checkpoints/epoch{finished_epochs}")
                
                

class PromptGenerator:
    def __init__(self, args, data_path):
        self.args = args
        self.prompts = {
            "general": self.load_prompt(args.general_prompt_path)
        }
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for key in MODEL_SPEAKER_TOKENS.keys():
            if key.lower() in self.args.model_name.lower():
                self.speaker_tokens = MODEL_SPEAKER_TOKENS[key]
                print(f"Using speaker tokens for {key}: {self.speaker_tokens}")
                break
        
        if self.args.fewshot_mode == "static" or self.args.fewshot_mode == "mix_static":
            self.fewshot_examples = self.set_static_fewshot(data)
        
        
    def load_prompt(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def fewshot_format(self, fewshot_examples, mode="fewshot"):
        text = f"카테고리: {fewshot_examples['input']['category']}\n도메인: {fewshot_examples['input']['domain']}\n문제 유형: {fewshot_examples['input']['question_type']}\n주제 키워드: {fewshot_examples['input']['topic_keyword']}\n질문: {fewshot_examples['input']['question']}"
        
        if mode == "fewshot":
            if "COT" in self.args.general_prompt_path:
                return f"{text}\n 생각 과정: (예시에서는 skip)\n 답변: {fewshot_examples['output']['answer']}"
            else:
                return f"{text}\n답변: {fewshot_examples['output']['answer']}"
        return text
        
    def set_static_fewshot(self, fewshot_examples):
        self.selection_data = []
        self.generation_data = []
        self.short_data = []
        for i in range(len(fewshot_examples)):
            if fewshot_examples[i]["input"]['question_type'] == '선다형':
                if len(self.selection_data) < self.args.num_fewshot:
                    self.selection_data.append(fewshot_examples[i])
            elif fewshot_examples[i]["input"]['question_type'] == '서술형':
                if len(self.generation_data) < self.args.num_fewshot:
                    self.generation_data.append(fewshot_examples[i])
            elif fewshot_examples[i]["input"]['question_type'] == '단답형':
                if len(self.short_data) < self.args.num_fewshot:
                    self.short_data.append(fewshot_examples[i])
            
            if len(self.selection_data) >= self.args.num_fewshot and \
               len(self.generation_data) >= self.args.num_fewshot and \
               len(self.short_data) >= self.args.num_fewshot:
                break
        
        if self.args.fewshot_mode == "mix_static":
            tmp_fewshot_text = []
            for i in range(self.args.num_fewshot):
                if i % 3 == 0:
                    tmp_fewshot_text.append(self.fewshot_format(self.selection_data[i // 3]))
                elif i % 3 == 1:
                    tmp_fewshot_text.append(self.fewshot_format(self.generation_data[i // 3]))
                elif i % 3 == 2:
                    tmp_fewshot_text.append(self.fewshot_format(self.short_data[i // 3]))
            
            fewshot_prompts = {
                "선다형": "예시:\n\n" + "\n".join(tmp_fewshot_text),
                "서술형": "예시:\n\n" + "\n".join(tmp_fewshot_text),
                "단답형": "예시:\n\n" + "\n".join(tmp_fewshot_text)
            }
            
        elif self.args.fewshot_mode == "static":
            fewshot_prompts = {}
            for dn, ds in zip(["선다형", "서술형", "단답형"], [self.selection_data, self.generation_data, self.short_data]):
                tmp_fewshot_text = []
                for i in range(len(ds)):
                    tmp_fewshot_text.append(self.fewshot_format(ds[i]))
                fewshot_prompts[dn] = "예시:\n\n"+"\n\n".join(tmp_fewshot_text)
        
        return fewshot_prompts
        
    def generate_prompt(self, type):
        if self.args.prompt_mode == "general":
            if self.args.fewshot_mode == "none":
                return self.prompts["general"]
            elif self.args.fewshot_mode == "static" or self.args.fewshot_mode == "mix_static":
                fewshot_text = self.fewshot_examples[type]
                return f"{self.prompts['general']}\n\n{fewshot_text}"
        

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, prompt_generator=None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.prompt_generator = prompt_generator

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.prompt_generator:
            prompt = self.prompt_generator.generate_prompt(item['input']['question_type'])
        else:
            prompt=""
        
        question_type = item['input']['question_type']
        question = item['input']['question']
        answer = item['output']['answer']
        if "#" in answer:
            answer = answer.split("#")[0].strip()
        
        #data for LLM(decoder only)
        # input_text = f"문제 유형: {question_type}\n질문: {question}\n답변:"
        input_text = self.prompt_generator.fewshot_format(item, mode="no_answer")
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ]
        inputs = self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        
        labels = self.tokenizer(answer).input_ids

        if labels[0] == self.tokenizer.bos_token_id:
            labels = labels[1:]

        return {
            "input_ids": inputs,
            "labels": labels
        }
    
    def collate_fn(self, batch):
        inputs = [torch.tensor(item["input_ids"] + item["labels"] + [self.tokenizer.eos_token_id]) for item in batch]
        labels = [torch.tensor([-100] * len(item["input_ids"]) + item["labels"] + [self.tokenizer.eos_token_id]) for item in batch]
        attention_mask = [torch.ones(len(input_ids), dtype=torch.long) for input_ids in inputs]
        
        input_ids = pad_sequence(inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class CustomDatasetForTest(CustomDataset):
    def __init__(self, data_path, tokenizer, prompt_generator=None):
        super().__init__(data_path, tokenizer, prompt_generator)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.prompt_generator:
            prompt = self.prompt_generator.generate_prompt(item['input']['question_type'])
        else:
            prompt=""
        
        #data for LLM(decoder only)
        # input_text = f"문제 유형: {question_type}\n질문: {question}\n답변:"
        input_text = self.prompt_generator.fewshot_format(item, mode="no_answer")
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ]
        inputs = self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        
        return {
            "input_ids": inputs[0],
            "attention_mask": torch.ones(len(inputs[0]), dtype=torch.long),
            "data": copy.deepcopy(item),  # Store the expected answer for evaluation
        }
        
    def collate_fn(self, batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        datas = [item["data"] for item in batch]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "data": datas
        }
        
        
def test(args, model, tokenizer, test_dataset, generate_args, post_process_fn=None):
    model.eval()
    total_correct = 0
    total_count = 0
    
    #copy input data json
    import copy
    test_data = copy.deepcopy(test_dataset.data)
    whole_data = copy.deepcopy(test_dataset.data)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    if len(test_data) == len(test_loader):
        print("Test dataset size matches DataLoader size.")
    
    print("Starting testing...")
    print(f"Total test samples: {len(test_loader)}")
    #no batch processing
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_args)[0]
            generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
            
            # Assuming the answer is in the 'answer' field of the test data
            if "output" in test_data[i].keys():
                expected_answer = test_data[i]["output"]["answer"]
            else:
                expected_answer = "IS NOT PROVIDED"
                test_data[i]["output"] = {}
                whole_data[i]["output"] = {}
                
            predicted_answer = generated_text.split("답변:")[-1].strip()
            
            test_data[i]["output"]["answer"] = predicted_answer
            whole_data[i]["output"]["gold"] = expected_answer
            whole_data[i]["output"]["answer"] = predicted_answer
            whole_data[i]["output"]["CoT_token"] = generated_text.split("assistant")[-1].strip()
            whole_data[i]["output"]["all_token"] = generated_text
            
            if post_process_fn:
                test_data[i] = post_process_fn(test_data[i])
                whole_data[i] = post_process_fn(whole_data[i])
                
            print(f"\n\nTest {i+1}/{len(test_loader)}: \nQuestion: {test_data[i]['input']['question']}")
            print(f"Expected: {expected_answer}\nPredicted: {test_data[i]['output']['answer']}\nCoT:{generated_text.split('assistant')[-1].strip()}")
    
    return test_data, whole_data
    
def parse_args():
    parser = argparse.ArgumentParser(description="Find model script")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to find")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on (default: cpu)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache the model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    
    # training configurations
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for training")
    parser.add_argument("--max_grad_norm", type=float, default=0.2, help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], help="Learning rate scheduler type")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--no_flash_attention", action="store_true", default=False, help="Disable flash attention")

    # lora configurations
    parser.add_argument("--lora_mode", type=str, default=None, choices=["lora"], help="LoRA mode to use")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="LoRA target modules")
    
    # RL configurations
    parser.add_argument("--rl_mode", type=str, default=None, choices=["ppo", "reinforce"], help="RL mode to use")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates to generate")
    parser.add_argument("--cand_temperature", type=float, default=0.6, help="Temperature for candidate generation")
    parser.add_argument("--cand_top_p", type=float, default=0.95, help="Top-p for candidate generation")
    parser.add_argument("--cand_repetition_penalty", type=float, default=1.0, help="Repetition penalty for candidate generation")
    # parser.add_argument("--ppo_clip_range", type=float, default=0.2, help="PPO clip range")
    # parser.add_argument("--kl_beta", type=float, default=0.0, help="KL divergence beta for PPO")
    parser.add_argument("--use_write_type_answer", action="store_true", default=False, help="Flag to use write type answer")
    parser.add_argument("--wta_reward_stretegy", type=str, default="cand_max", choices=["cand_max", "cand_diff", "cand_include", "1"], help="Strategy for write type answer reward")
    parser.add_argument("--use_format_reward", action="store_true", default=False, help="Flag to use format reward")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Scale for the reward function")
    parser.add_argument("--generation_reward_scale", type=float, default=1.0, help="Scale for the generation reward function")
    
    # data configurations
    parser.add_argument("--train_data_path", type=str, default="data/QA/korean_culture_qa_V1.0_train+.json", help="Path to training data for custom dataset")
    parser.add_argument("--dev_data_path", type=str, default="data/QA/korean_culture_qa_V1.0_dev+.json", help="Path to prompt generator script")
    parser.add_argument("--test_data_path", type=str, default="data/QA/korean_culture_qa_V1.0_test+.json", help="Path to test data for custom dataset")
    parser.add_argument("--general_prompt_path", type=str, default="prompts/COT공용프롬프트.txt", help="Path to generation prompt script")
    
    parser.add_argument("--prompt_mode", type=str, default="general", choices=["general"], help="Mode for prompt generation")
    parser.add_argument("--fewshot_mode", type=str, default="static", choices=["none", "mix_static", "static"], help="Mode for few-shot learning")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples to use")
    
    # mode configurations
    parser.add_argument("--train", action="store_true", default=False, help="Flag to indicate if training should be performed")
    parser.add_argument("--test", action="store_true", default=False, help="Flag to indicate if testing should be performed") 
    parser.add_argument("--testing_every_epoch", type=int, default=1, help="Number of epochs after which to test the model")
    
    parser.add_argument("--run_name", type=str, default="train", help="Name of the run for logging purposes")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    args.run_name = f"{args.run_name}_{args.model_name.replace('/', '_')}_fewshot_{args.fewshot_mode}_num_fewshot_{args.num_fewshot}"
    print(f"Run name set to: {args.run_name}")
    
    if args.rl_mode:
        args.run_name = f"{args.run_name}_rl_mode_{args.rl_mode}_n_cand_{args.num_candidates}_c_temp_{args.cand_temperature}_c_tp_{args.cand_top_p}_c_rp_{args.cand_repetition_penalty}_rs_{args.reward_scale}_grs_{args.generation_reward_scale}"
        if args.use_write_type_answer:
            args.run_name += f"_wta_{args.wta_reward_stretegy}"
        if args.use_format_reward:
            args.run_name += "_fr"
        print(f"Run name updated for RL: {args.run_name}")
        
    if args.lora_mode:
        args.run_name = f"{args.run_name}_lora_r{args.lora_r}_a{args.lora_alpha}_dp{args.lora_dropout}"
        print(f"Run name updated for LoRA: {args.run_name}")
    
    if args.train:
        args.run_name = f"train/{args.run_name}"
    else:
        args.run_name = f"basemodel/{args.run_name}"
    
    save_dir = f"results/{args.run_name}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/args.json", "w") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # 4‑bit 가중치
        bnb_4bit_quant_type="nf4",               # Normal‑Float 4
        bnb_4bit_use_double_quant=True,          # double‑quant
        bnb_4bit_compute_dtype=torch.bfloat16,   # Ada, Hopper, MI300 등
        llm_int8_skip_modules=["lm_head"]        # 출력층은 FP16
    )
    
    if args.no_flash_attention:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                        device_map=args.device, 
                                                        cache_dir=args.cache_dir, 
                                                        trust_remote_code=True,
                                                        quantization_config=bnb_config,     # ★ QLoRA 핵심
                                                        # attn_implementation="flash_attention_2",  # Flash Attention 2
                                                        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                        device_map=args.device, 
                                                        cache_dir=args.cache_dir, 
                                                        trust_remote_code=True,
                                                        quantization_config=bnb_config,     # ★ QLoRA 핵심
                                                        attn_implementation="flash_attention_2",  # Flash Attention
                                                        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    generation_config= GenerationConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    
    # if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to EOS token: {tokenizer.pad_token}")
    
    if args.lora_mode == "lora":
        model = prepare_model_for_kbit_training(model)  # Prepare model for k-bit training
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.lora_target_modules.split(",")
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("LoRA model loaded successfully.")
    
    # if args.train:
    #     model.gradient_checkpointing_enable()  # it enables in prepare_model_for_kbit_training
        
    print(f"Model '{args.model_name}' found and loaded successfully.")
    
    
    prompt_generator = PromptGenerator(args, args.train_data_path)
    if args.rl_mode is None:
        train_dataset = CustomDataset(args.train_data_path, tokenizer, prompt_generator)
        dev_dataset = CustomDataset(args.dev_data_path, tokenizer, prompt_generator)
    else:
        train_dataset = CustomDatasetForTest(args.train_data_path, tokenizer, prompt_generator)
        dev_dataset = CustomDatasetForTest(args.dev_data_path, tokenizer, prompt_generator)
    dev_dataset_for_test = CustomDatasetForTest(args.dev_data_path, tokenizer, prompt_generator)
    test_dataset = CustomDatasetForTest(args.test_data_path, tokenizer, prompt_generator)
    
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    post_process_fn = None
    for key in MODEL_NAME_to_POST_FUNCTION.keys():
        if key.lower() in args.model_name.lower():
            post_process_fn = MODEL_NAME_to_POST_FUNCTION[key]
            print(f"Using post-processing function: {post_process_fn.__name__}") 
    
    train_generate_args = GenerationConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir).to_dict()
    train_generate_args["max_new_tokens"] = 2048  
    train_generate_args["do_sample"] = True
    train_generate_args["temperature"] = args.cand_temperature #1.0
    train_generate_args["top_p"] = args.cand_top_p #0.95
    train_generate_args["repetition_penalty"] = args.cand_repetition_penalty #1.0
    train_generate_args["num_return_sequences"] = args.num_candidates # 5
    print(f"Train generation args: {train_generate_args}")
    
    test_generate_args = GenerationConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir).to_dict()
    test_generate_args["max_new_tokens"] = 2048
    test_generate_args["num_beams"] = 5
    test_generate_args["do_sample"] = False
    test_generate_args["top_p"] = args.cand_top_p
    test_generate_args["temperature"] = args.cand_temperature
    print(f"Test generation args: {test_generate_args}")
    
    if "hyperclovax" in args.model_name.lower():
        train_generate_args["use_cache"] = True
        train_generate_args["stop_strings"] = ["<|endofturn|>", "<|stop|>", "<|im_end|>"]
        train_generate_args["tokenizer"] = tokenizer
        
        test_generate_args["use_cache"] = True
        test_generate_args["stop_strings"] = ["<|endofturn|>", "<|stop|>", "<|im_end|>"]
        test_generate_args["tokenizer"] = tokenizer
    
    if args.train:
        
        RL_model_instance = None
        no_dev = None
        if args.rl_mode:
            RL_model_instance = RL_model(model, tokenizer, train_generate_args, args)
            no_dev = True
            
        if args.testing_every_epoch is not None:
            print(f"Testing will be performed every {args.testing_every_epoch} epochs.")
            callback = CustomCallback(
                args=args,
                model=model,
                tokenizer=tokenizer,
                test_fn=test,
                gen_args=test_generate_args,
                post_process_fn=post_process_fn,
                dev_dataset=dev_dataset_for_test,  # Will be set later
                test_dataset=test_dataset,  # Will be set later
                RL_model=RL_model_instance  # Will be set later if RL mode is used
            )
        
        training_args = TrainingArguments(
            output_dir=save_dir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            logging_dir="logs",
            logging_steps=10,
            save_strategy="epoch",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_steps=None if no_dev else 1000,
            bf16=torch.cuda.is_available(),  # Hopper/Ada 권장
            fp16=not torch.cuda.is_bf16_supported(),  # fallback
            remove_unused_columns=False,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
        )
            
        trainer = Trainer(
            model=RL_model_instance if RL_model_instance else model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None if no_dev else dev_dataset,
            data_collator=train_dataset.collate_fn,
            tokenizer=tokenizer,
            callbacks=[callback] if args.testing_every_epoch is not None else [],
        )

        print("Starting training...")
        trainer.train()
        # trainer.save_model("output/final_model")
        # tokenizer.save_pretrained("output/final_model")
        print("Training completed and model saved.")
        
if __name__ == "__main__":
    main()