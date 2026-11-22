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

        system_prompt = """
        You are a teacher for an entity matching AI model.

        Entity matching is the process of matching two entries in different databases that are functionally identical despite having different names or attributes, due to natural human error or differences in data storage.
        A Large Language Model is tasked with finding these matches by performing entity matching; however, it will not perform perfectly.
        In order to train this matching model, it must be provided with more examples in areas it has diffculty.

        You will be provided with a selection of examples that a learner model failed to match correctly.
        Generate a similar but not same labeled entity matching example based on the failed examples.

        Use DITTO-format when producing your example.

        DITTO-format: one entity looks like: "[COL] attr1 [VAL] val1 · · · [COL] attr𝑝 [VAL] val𝑝"
                              one sample looks like: left_entity \t right_entity \t label(0 or 1)
                              example: "COL title VAL "ASUS GeForce RTX 2060 SUPER TURBO EVO"@en	COL title VAL "Gigabyte Aorus GeForce RTX 2060 Super 8GB Graphics Card"@en 0"
        """

        formatted = []

        # assumes a tuple format for selected_examples. ()
        for i, example in enumerate(selected_examples, start=1):
            left, right, label = example
            formatted.append(
                f"Example {i}:\n"
                f"Left entity: {left}\n"
                f"Right entity: {right}\n"
                f"Label: {label} "
                f"({'match' if label == 1 else 'no match'})"
            )

        formatted_examples = "\n\n".join(formatted)

        # Not sure how we want to specify to generate positive or negative examples! Random?
        user_prompt = f"""
        Generate a new pair of entities that match, with the correct format, based on the following examples:

        {formatted_examples}
        """

        for _ in range(num_examples):
            # Left max_new_tokens and temperature at default
            teacher_output.append(self._generate(user_prompt, system_prompt))

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
        