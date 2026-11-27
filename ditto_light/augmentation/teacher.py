# teacher.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random

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
        selected_examples = [ex for ex in selected_examples if ex[2] == 1]

        if not selected_examples:
            return teacher_output

        system_prompt = """
        You generate hard negative examples for entity matching.

        You will receive two records referring to the same real-world entity.

        Create a modified version of the RIGHT record that refers to a different but similar real-world entity.

        Requirements:
        - Ensure that at least one identity-defining attribute changes substantially, not just its spelling or formatting.
        - For restaurants, changing only cuisine is insufficient.
        - For products, changing only descriptive text is insufficient.
        - For publications, changing only title formatting is insufficient.
        - Ensure the modified attributes are internally consistent.

        For example, changing a restaurant's cuisine alone does not establish a different restaurant. Changing a product's description alone does not establish a different product.

        Return ONLY the modified right record on a single line. Do not return the left record, a label, or an explanation.
        """

        for i in range(num_examples):
            left, right, label = selected_examples[
                i % len(selected_examples)
            ]

            user_prompt = (
                f"Original left record:\n{left}\n\n"
                f"Original right record:\n{right}\n\n"
                "Generate a modified right record representing "
                "a different but similar entity."
            )

            modified_right = self._generate(
                user_prompt, system_prompt
            ).strip()

            teacher_output.append(
                (left, right, modified_right)
            )
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
        seen = set()

        for left, original_right, modified_right in teacher_outputs:
            if not modified_right:
                continue

            # Require a single DITTO-formatted entity
            if "\n" in modified_right or "\t" in modified_right:
                continue

            if not modified_right.startswith("COL "):
                continue

            if " VAL " not in modified_right:
                continue

            # Reject unchanged entities
            if modified_right.lower() == original_right.lower():
                continue

            if modified_right.lower() == left.lower():
                continue

            # Remove duplicates
            key = (left.lower(), modified_right.lower())

            if key in seen:
                continue

            seen.add(key)

            # Assign negative label deterministically
            new_augmented_examples.append(
                (left, modified_right, 0)
            )

        return new_augmented_examples
        