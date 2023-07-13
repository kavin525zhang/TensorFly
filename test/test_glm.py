from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("/data9/NFS/patent/model_hub/chatglm2-6b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("/data9/NFS/patent/model_hub/chatglm2-6b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

inputs = tokenizer(
    ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
    return_tensors="pt", padding=True)
inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
inputs = inputs.to('cuda')
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
print("loss:", loss)
print("logits:", logits)
