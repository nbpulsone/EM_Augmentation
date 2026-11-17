# teacher.py

class TeacherModel:

    def __init__(self, model_name):
        # add support for loading whatever models we want to use
        # should probably research what would be a good open source LLM to use as teacher model
        self.model_name = model_name
        self.model = None 
        # TODO: load a good teacher model


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
        