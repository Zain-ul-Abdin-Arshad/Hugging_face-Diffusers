from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
prompt = "A genius student at University of Lahore."
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(
    input_ids,
    max_length=200,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    early_stopping=True,
    attention_mask=input_ids.ne(tokenizer.pad_token_id),
    pad_token_id=tokenizer.pad_token_id
)
generated_story = tokenizer.decode(output[0], skip_special_tokens=False)
print("Generated Story:\n")
print(generated_story)