from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, TrainerCallback, GenerationConfig, TextStreamer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModelForCausalLM
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

class PromptGenerator:
    def __init__(self, args, data_path):
        self.args = args
        self.prompts = {
            "general": self.load_prompt(args.general_prompt_path),
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
            return f"{text}\n 생각 과정: (예시에서는 skip)\n 답변: {fewshot_examples['output']['answer']}"
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
        self.original_chat_template = copy.deepcopy(tokenizer.chat_template)

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
        
        #data for LLM(decoder only)
        # input_text = f"문제 유형: {question_type}\n질문: {question}\n답변:"
        input_text = self.prompt_generator.fewshot_format(item, mode="no_answer")
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ]
        
        
        inputs = self.tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        
        
        labels = self.tokenizer(answer).input_ids
        
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
        
        question_type = item['input']['question_type']
        
        if question_type == "선다형":
            self.tokenizer.chat_template = self.original_chat_template.replace("strftime_now('%d %b %Y')", "'28 Jul 2025'")  # Set a fixed date for reproducibility
        else:
            self.tokenizer.chat_template = self.original_chat_template.replace("strftime_now('%d %b %Y')", "'29 Jul 2025'")  # Set a fixed date for reproducibility
            
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
    
    print(model)
    
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
            
            if batch["data"]["input"]["question_type"][0] == "서술형":
                generate_args["num_beams"] = 5
                model.set_adapter("default")
            elif batch["data"]["input"]["question_type"][0] == "단답형":
                generate_args["num_beams"] = 5
                model.set_adapter("default")
            elif batch["data"]["input"]["question_type"][0] == "선다형":
                generate_args["num_beams"] = 1
                model.set_adapter("selection")
            else:
                print("error")
            print(generate_args)
            
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
            with open(f"{args.run_name}/tmp.json", 'w', encoding='utf-8') as f:
                json.dump(whole_data, f, ensure_ascii=False, indent=4)
    
    return test_data, whole_data
    
def parse_args():
    parser = argparse.ArgumentParser(description="Find model script")
    parser.add_argument("--adapter_path", type=str, default=None, required=True, help="Path to the model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device to load the model on (default: cpu)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache the model")
    
    # data configurations
    parser.add_argument("--train_data_path", type=str, default="data/QA/korean_culture_qa_V1.0_train+.json", help="Path to training data for custom dataset")
    parser.add_argument("--dev_data_path", type=str, default="data/QA/korean_culture_qa_V1.0_dev+.json", help="Path to prompt generator script")
    parser.add_argument("--test_data_path", type=str, default="data/QA/korean_culture_qa_V1.0_test+.json", help="Path to test data for custom dataset")
    parser.add_argument("--general_prompt_path", type=str, default="prompts/COT공용프롬프트.txt", help="Path to generation prompt script")
    
    parser.add_argument("--prompt_mode", type=str, default="general", choices=["general"], help="Mode for prompt generation")
    parser.add_argument("--fewshot_mode", type=str, default="static", choices=["none", "mix_static", "static"], help="Mode for few-shot learning")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples to use")

    parser.add_argument("--run_name", type=str, default="reproduce_result", help="Name of the run for logging purposes")

    return parser.parse_args()


def main():
    args = parse_args()
    lora_config = LoraConfig.from_pretrained(args.adapter_path)
    args.model_name = lora_config.base_model_name_or_path
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                       # 4‑bit 가중치
        bnb_4bit_quant_type="nf4",               # Normal‑Float 4
        bnb_4bit_use_double_quant=True,          # double‑quant
        bnb_4bit_compute_dtype=torch.bfloat16,   # Ada, Hopper, MI300 등
        llm_int8_skip_modules=["lm_head"]        # 출력층은 FP16
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                    device_map=args.device, 
                                                    cache_dir=args.cache_dir, 
                                                    trust_remote_code=True,
                                                    quantization_config=bnb_config,     # ★ QLoRA 핵심
                                                    attn_implementation="flash_attention_2",  # Flash Attention 2
                                                    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    generation_config= GenerationConfig.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    
    base_model = prepare_model_for_kbit_training(base_model)
    model = PeftModelForCausalLM.from_pretrained(base_model, args.adapter_path)
    model.load_adapter(args.adapter_path, subfolder="selection", adapter_name="selection")
    
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token set to EOS token: {tokenizer.pad_token}")
    
    prompt_generator = PromptGenerator(args, args.train_data_path)
    dev_dataset_for_test = CustomDatasetForTest(args.dev_data_path, tokenizer, prompt_generator)
    test_dataset = CustomDatasetForTest(args.test_data_path, tokenizer, prompt_generator)
    
    print(f"Dev dataset size: {len(dev_dataset_for_test)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    post_process_fn = None
    for key in MODEL_NAME_to_POST_FUNCTION.keys():
        if key.lower() in args.model_name.lower():
            post_process_fn = MODEL_NAME_to_POST_FUNCTION[key]
            print(f"Using post-processing function: {post_process_fn.__name__}") 
            
    generate_args = {
        "max_new_tokens": 1024,
        "num_beams": 5,
        "do_sample": False,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 1.0
    }

    save_path = f"{args.run_name}"
    
    print("Starting testing...")
    os.makedirs(save_path, exist_ok=True)
    dev_results, dev_whole_data = test(args, model, tokenizer, dev_dataset_for_test, generate_args, post_process_fn)
    with open(f"{save_path}/dev_results.json", 'w', encoding='utf-8') as f:
        json.dump(dev_results, f, ensure_ascii=False, indent=4)
    with open(f"{save_path}/dev_whole_data.json", 'w', encoding='utf-8') as f:
        json.dump(dev_whole_data, f, ensure_ascii=False, indent=4)
        
    dev_score = evaluation_korean_contest_culture_QA(dev_dataset_for_test.data, dev_results)
    with open(f"{save_path}/dev_score.json", 'w', encoding='utf-8') as f:
        json.dump(dev_score, f, ensure_ascii=False, indent=4)
        
    # print(f"Testing dev dataset completed and results saved. \nScore: {dev_score}\n\n============================\n")
    
    test_results, test_whole_data = test(args, model, tokenizer, test_dataset, generate_args, post_process_fn)
    with open(f"{save_path}/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    with open(f"{save_path}/test_whole_data.json", 'w', encoding='utf-8') as f:
        json.dump(test_whole_data, f, ensure_ascii=False, indent=4)
        
    print(f"Testing test dataset completed and results saved.")
    
    
        
if __name__ == "__main__":
    main()