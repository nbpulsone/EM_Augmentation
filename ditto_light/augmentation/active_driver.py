from .error_selector import select_examples
from .teacher import TeacherModel

class ActiveAugmentationDriver:

    def __init__(self, hp):
        self.hp = hp
        self.teacher = TeacherModel(hp.teacher_lm)

    def run_round(self, train_dataset, feedback_dataset, evaluation_details):

        # find informative validation examples from student weaknesses
        selected = select_examples(feedback_dataset, evaluation_details, self.hp)

        # teacher creates targeted examples
        teacher_outputs = self.teacher.generate_examples(selected,self.hp.aug_examples)

        # convert to valid EM training examples
        augmented_examples = self.teacher.build_augmented_examples(teacher_outputs)

        # update training data with new augmenets
        train_dataset.add_new_examples(augmented_examples)

        return augmented_examples
