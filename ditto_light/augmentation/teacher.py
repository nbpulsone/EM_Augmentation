# teacher.py

class TeacherModel:

    def __init__(self, model_name):
        # add support for loading whatever models we want to use
        # should probably research what would be a good open source LLM to use as teacher model
        self.model_name = model_name
        self.model = None 
        # TODO


    def generate_examples(self, selected_examples, num_examples):
        """
        generate new labeled entity-matching examples targeted at weaknesses identified in the student model

        returns: model_output
        """
        model_output = None

        # TODO

        return model_output

    def build_augmented_examples(self, teacher_outputs):
        """
        convert teacher output into DITTO-compatible examples

        format: fuse two entities, e.g. entity = "[COL] attr1 [VAL] val1 · · · [COL] attr𝑝 [VAL] val𝑝"
                one sample: left_entity \t right_entity \t label(0 or 1)
                e.g. "COL title VAL "ASUS GeForce RTX 2060 SUPER TURBO EVO"@en	COL title VAL "Gigabyte Aorus GeForce RTX 2060 Super 8GB Graphics Card"@en 0"

        maybe remove duplicates, enforce postivie/negative balance, etc.
        
        returns: new_augmented_examples
        """
        new_augmented_examples = []

        # TODO

        return new_augmented_examples
        