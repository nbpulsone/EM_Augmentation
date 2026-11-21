# teacher.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class TeacherModel:

    DEFAULT_TEACHER = "Qwen/Qwen2.5-7B-Instruct"

    def __init__(self, model_name=DEFAULT_TEACHER, load_in_4bit=True, device_map="auto"):
        # add support for loading whatever models we want to use
        # should probably research what would be a good open source LLM to use as teacher model

        # TODO: load a good teacher model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        quant_config = None
        if load_in_4bit and torch.cuda.is_available():
            # cuts memory ~4x with small quality loss; needs `pip install bitsandbytes`
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )
        self.model.eval()

    def _generate(self, prompt, system_prompt, max_new_tokens=512, temperature=0.8):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


    def generate_examples(self, selected_examples, num_examples):
        """
        prompt the teacher model to generate new labeled entity-matching examples 
        
        include selected_examples in the teacher prompt to inform its augmentation response

        returns: teacher_output (list of responses of size "num_examples")
        """
        teacher_output = []

        # TODO: prompt teacher model "num_examples" times

        return teacher_output

    def build_augmented_examples(self, teacher_outputs):
        """
        convert teacher outputs into DITTO-compatible examples

        DITTO-format: one entity looks like: "[COL] attr1 [VAL] val1 · · · [COL] attr𝑝 [VAL] val𝑝"
                      one sample looks like: left_entity \t right_entity \t label(0 or 1)
                      example: "COL title VAL "ASUS GeForce RTX 2060 SUPER TURBO EVO"@en	COL title VAL "Gigabyte Aorus GeForce RTX 2060 Super 8GB Graphics Card"@en 0"

        maybe remove duplicates, enforce postivie/negative balance, etc.
        
        returns: new_augmented_examples
        """
        new_augmented_examples = []

        # TODO: parse teacher outputs to extract new augmented examples

        return new_augmented_examples
        