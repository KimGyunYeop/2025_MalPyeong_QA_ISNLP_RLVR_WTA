def post_processing_general(data):
    if data["input"]["question_type"] == "선다형":
        for i in range(1, 5):
            if str(i) in data["output"]["answer"]:
                data["output"]["answer"] = str(i)
    
    
    if data["input"]["question_type"] == "서술형":
        while True:
            if "**" in data["output"]["answer"]:
                data["output"]["answer"] = data["output"]["answer"].replace("*", "")
            elif "\n" in data["output"]["answer"]:
                data["output"]["answer"] = data["output"]["answer"].replace("\n", " ")
            else:
                break
                
    return data

def post_processing_exaone(data):
    data = post_processing_general(data)
    
    if "[|assistant|]" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].replace("[|assistant|]", "").strip()
    
    for b,a in zip(["ᄀ", "ᄂ", "ᄃ", "ᄅ"], ["ㄱ", "ㄴ", "ㄷ", "ㄹ"]):
        if b in data["output"]["answer"]:
            data["output"]["answer"] = data["output"]["answer"].replace(b, a)
    
        
    return data

def post_processing_kanana(data):
    if "assistant" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].replace("assistant", "").strip()
        
    data = post_processing_general(data)
    
        
    return data

def post_processing_midm(data):
    if "assistant" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].split("assistant")[-1].strip()
        
    data = post_processing_general(data)
    
        
    return data

def post_processing_ax(data):
    if "assistant" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].split("assistant")[-1].strip()
        
    data = post_processing_general(data)
    
        
    return data


def post_processing_tri(data):
    # delete "assistant" in the answer
    if "assistant" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].split("assistant")[-1].strip()
        
    if "**답변:**" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].split("**답변:**")[-1].strip()
        
    data["output"]["answer"] = data["output"]["answer"].replace("**", "").strip()
    
    if "answer:" in data["output"]["answer"]:
        data["output"]["answer"] = data["output"]["answer"].split("answer:")[-1].strip()
    
    data = post_processing_general(data)

    return data

MODEL_NAME_to_POST_FUNCTION = {
    "EXAONE" : post_processing_exaone,
    "kanana" : post_processing_kanana,
    "Midm" : post_processing_midm,
    "Tri" : post_processing_tri,  # Assuming Tri uses the same post-processing as Midm
    "A.X" : post_processing_ax,
    "hyperclovax" : post_processing_midm,
}

MODEL_SPEAKER_TOKENS = {
    "EXAONE" : {
        "assistant": "[|assistant|]",
        "user": "[|user|]",
    },
    "kanana" : {
        "assistant": "assistant",
        "user": "user",
    },
    "Midm" : {
        "assistant": "assistant",
        "user": "user",
    },
    "Tri" : {
        "assistant": "assistant",
        "user": "user",
    },
    "A.X" : {
        "assistant": "assistant",
        "user": "user",
    },
    "hyperclovax" : {
        "assistant": "assistant",
        "user": "user",
    },
}