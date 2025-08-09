from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu
from rouge_metric import Rouge
import random
from konlpy.tag import Mecab
import evaluate
# import tensorflow as tf

# GPU 메모리 증가를 허용하도록 설정
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


bert_scorer = evaluate.load('bertscore')
bert_model_type = 'bert-base-multilingual-cased'

# from bleurt import score

checkpoint = "BLEURT-20"

# scorer = score.BleurtScorer(checkpoint)

tokenizer = Mecab()

def calc_formatting_score(pred_data):
    correct = 0
    total = len(pred_data)

    for pred in pred_data:
        if "답변:" in pred["output"]["CoT_token"]:
            if len(pred["output"]["CoT_token"].split("답변:")[0].replace("생각 과정:","")) > 3:
                correct += 1

    return correct / total if total > 0 else 0

def calc_soft_exact_match(true_data, pred_data):
    correct = 0
    total = len(true_data)
    
    for true, pred in zip(true_data, pred_data):
        acceptable_answers = true.split('#')
        for ans in acceptable_answers:
            if ans.strip() in pred.strip():
                correct += 1
                break
                
            if pred.strip() in ans.strip():
                correct += 1
                break
            
    return correct / total if total > 0 else 0

def expanded_evaluation_korean_contest_culture_QA(true_data, pred_data):
    # Separate questions by type
    multiple_choice_qs = {"true": [], "pred": []}
    short_answer_qs = {"true": [], "pred": []}
    descriptive_qs = {"true": [], "pred": []}
    
    multiple_pred_datas = []
    short_pred_datas = []
    descriptive_pred_datas = []

    # Categorize questions by type
    for true_item, pred_item in zip(true_data, pred_data):
        if true_item["id"] != pred_item["id"]:
            return {
                "error": f"ID mismatch: {true_item['id']} != {pred_item['id']}"
            }
            
        q_type = true_item["input"]["question_type"]
        true_ans = true_item["output"]["answer"]
        pred_ans = pred_item["output"]["answer"]
        
        if q_type == "선다형":
            multiple_choice_qs["true"].append(true_ans)
            multiple_choice_qs["pred"].append(pred_ans)
            multiple_pred_datas.append(pred_item)
        elif q_type == "단답형":
            short_answer_qs["true"].append(true_ans)
            short_answer_qs["pred"].append(pred_ans)
            short_pred_datas.append(pred_item)
        elif q_type == "서술형":
            descriptive_qs["true"].append(true_ans)
            descriptive_qs["pred"].append(pred_ans)
            descriptive_pred_datas.append(pred_item)
            
    # Calculate scores for each type
    scores = {}
    
    
    # Multiple choice questions (Accuracy)
    if multiple_choice_qs["true"]:
        scores["accuracy"] = calc_Accuracy(multiple_choice_qs["true"], multiple_choice_qs["pred"])
    else:
        scores["accuracy"] = 0
        
    # Short answer questions (Exact Match)
    if short_answer_qs["true"]:
        scores["exact_match"] = calc_exact_match(short_answer_qs["true"], short_answer_qs["pred"])
        scores["soft_exact_match"] = calc_soft_exact_match(short_answer_qs["true"], short_answer_qs["pred"])
    else:
        scores["exact_match"] = 0
        
    # Descriptive questions (ROUGE, BERTScore, BLEURT)
    if descriptive_qs["true"]:
        scores["rouge_1"] = calc_ROUGE_1(descriptive_qs["true"], descriptive_qs["pred"])
        scores["bertscore"] = calc_bertscore(descriptive_qs["true"], descriptive_qs["pred"])
        # scores["bleurt"] = calc_bleurt(descriptive_qs["true"], descriptive_qs["pred"])
        scores["descriptive_avg"] = (scores["rouge_1"] + scores["bertscore"]) / 2
    else:
        scores["rouge_1"] = 0
        scores["bertscore"] = 0
        # scores["bleurt"] = 0
        scores["descriptive_avg"] = 0

    scores["formatting_score_multi_short"] = calc_formatting_score(multiple_pred_datas + short_pred_datas)
    scores["formatting_score"] = calc_formatting_score(pred_data)

    # Calculate final score (average of the three types)
    type_scores = []
    if multiple_choice_qs["true"]:
        type_scores.append(scores["accuracy"])
    if short_answer_qs["true"]:
        type_scores.append(scores["exact_match"])
    if descriptive_qs["true"]:
        type_scores.append(scores["descriptive_avg"])
        
    scores["final_score"] = sum(type_scores) / len(type_scores) if type_scores else 0
    
    return scores

def evaluation_korean_contest_culture_QA(true_data, pred_data):
    # Separate questions by type
    multiple_choice_qs = {"true": [], "pred": []}
    short_answer_qs = {"true": [], "pred": []}
    descriptive_qs = {"true": [], "pred": []}
    
    # Categorize questions by type
    for true_item, pred_item in zip(true_data, pred_data):
        if true_item["id"] != pred_item["id"]:
            return {
                "error": f"ID mismatch: {true_item['id']} != {pred_item['id']}"
            }
            
        q_type = true_item["input"]["question_type"]
        true_ans = true_item["output"]["answer"]
        pred_ans = pred_item["output"]["answer"]
        
        if q_type == "선다형":
            multiple_choice_qs["true"].append(true_ans)
            multiple_choice_qs["pred"].append(pred_ans)
        elif q_type == "단답형":
            short_answer_qs["true"].append(true_ans)
            short_answer_qs["pred"].append(pred_ans)
        elif q_type == "서술형":
            descriptive_qs["true"].append(true_ans)
            descriptive_qs["pred"].append(pred_ans)
            
    # Calculate scores for each type
    scores = {}
    
    # Multiple choice questions (Accuracy)
    if multiple_choice_qs["true"]:
        scores["accuracy"] = calc_Accuracy(multiple_choice_qs["true"], multiple_choice_qs["pred"])
    else:
        scores["accuracy"] = 0
        
    # Short answer questions (Exact Match)
    if short_answer_qs["true"]:
        scores["exact_match"] = calc_exact_match(short_answer_qs["true"], short_answer_qs["pred"])
    else:
        scores["exact_match"] = 0
        
    # Descriptive questions (ROUGE, BERTScore, BLEURT)
    if descriptive_qs["true"]:
        scores["rouge_1"] = calc_ROUGE_1(descriptive_qs["true"], descriptive_qs["pred"])
        scores["bertscore"] = calc_bertscore(descriptive_qs["true"], descriptive_qs["pred"])
        # scores["bleurt"] = calc_bleurt(descriptive_qs["true"], descriptive_qs["pred"])
        scores["descriptive_avg"] = (scores["rouge_1"] + scores["bertscore"]) / 2
    else:
        scores["rouge_1"] = 0
        scores["bertscore"] = 0
        # scores["bleurt"] = 0
        scores["descriptive_avg"] = 0
        
    # Calculate final score (average of the three types)
    type_scores = []
    if multiple_choice_qs["true"]:
        type_scores.append(scores["accuracy"])
    if short_answer_qs["true"]:
        type_scores.append(scores["exact_match"])
    if descriptive_qs["true"]:
        type_scores.append(scores["descriptive_avg"])
        
    scores["final_score"] = sum(type_scores) / len(type_scores) if type_scores else 0
    
    return scores

def calc_Accuracy(true_data, pred_data):

    return accuracy_score(true_data, pred_data)

def calc_bertscore(true_data, pred_data):
    if type(true_data[0]) is list:
        true_data = list(map(lambda x: x[0], true_data))

    scores = bert_scorer.compute(predictions=pred_data, references=true_data, model_type=bert_model_type)

    return sum(scores['f1']) / len(scores['f1'])

def calc_exact_match(true_data, pred_data):
    """
    Calculate Exact Match score where true_data may contain multiple acceptable answers separated by #
    """
    correct = 0
    total = len(true_data)
    
    for true, pred in zip(true_data, pred_data):
        # Split true answer into acceptable variations
        acceptable_answers = true.split('#')
        # Check if prediction matches any acceptable answer
        if any(pred.strip() == ans.strip() for ans in acceptable_answers):
            correct += 1
            
    return correct / total if total > 0 else 0

def calc_ROUGE_1(true, pred):
    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.0,
    )

    scores = rouge_evaluator.get_scores(pred, true)
    return scores['rouge-1']['f']


def calc_ROUGE_L(true, pred):
    rouge_evaluator = Rouge(
        metrics=["rouge-n", "rouge-l"],
        max_n=2,
        limit_length=True,
        length_limit=1000,
        length_limit_type="words",
        use_tokenizer=True,
        apply_avg=True,
        apply_best=False,
        alpha=0.5,  # Default F1_score
        weight_factor=1.0,
    )

    scores = rouge_evaluator.get_scores(pred, true)
    return scores['rouge-l']['f']

def calc_BLEU(true, pred, apply_avg=True, apply_best=False, use_mecab=True):
    stacked_bleu = []

    if type(true[0]) is str:
        true = list(map(lambda x: [x], true))

    for i in range(len(true)):
        best_bleu = 0
        sum_bleu = 0
        for j in range(len(true[i])):

            if use_mecab:
                ref = tokenizer.morphs(true[i][j])
                candi = tokenizer.morphs(pred[i])
            else:
                ref = true[i][j].split()
                candi = pred[i].split()


            score = sentence_bleu([ref], candi, weights=(1, 0, 0, 0))

            sum_bleu += score
            if score > best_bleu:
                best_bleu = score

        avg_bleu = sum_bleu / len(true[i])
        if apply_best:
            stacked_bleu.append(best_bleu)
        if apply_avg:
            stacked_bleu.append(avg_bleu)

    return sum(stacked_bleu) / len(stacked_bleu)

# def calc_bleurt(true_data, pred_data):
#     if type(true_data[0]) is list:
#         true_data = list(map(lambda x: x[0], true_data))

#     scores = scorer.score(references=true_data, candidates=pred_data, batch_size=64)

#     return sum(scores) / len(scores)