"""HIR question builder (HIR repository, verl/trainer/ppo/ray_trainer.py, `_concat_question`); only the docstring is translated."""

class HIRQuestionBuilder:
    def _concat_question(self, question_raw, criteria, q_type):
        """Join a question and its criteria in one of HIR's six layouts.

        type1: criteria appended as sentences after the question, separated by a space
        type2: question, a newline, then the criteria as sentences
        type3: question, a blank line, "The output must follow the following rules:" and a numbered list
        type4: question, a blank line, then a "- " bullet list
        type5: criteria as sentences first, a blank line, then the question
        type6: question, a blank line, then one criterion per line
        """

        if q_type == "type1":
            instructions = ""
            for cc in criteria:
                assert len(cc) > 0
                if cc[-1] != "," and cc[-1] != "." and cc[-1] != "?" and cc[-1] != "!":
                    instructions += cc + ". "
                else:
                    instructions += cc + " "

            question = f"{question_raw} {instructions.strip()}"
        elif q_type == "type2":
            instructions = ""
            for cc in criteria:
                assert len(cc) > 0
                if cc[-1] != "," and cc[-1] != "." and cc[-1] != "?" and cc[-1] != "!":
                    instructions += cc + ". "
                else:
                    instructions += cc + " "
            question = f"{question_raw}\n{instructions.strip()}"
        elif q_type == "type3":
            instructions = "The output must follow the following rules:\n"
            for idx, cc in enumerate(criteria):
                assert len(cc) > 0
                instructions += f"{idx+1}. " + cc + "\n"

            question = f"{question_raw}\n\n{instructions.strip()}"
        elif q_type == "type4":
            instructions = ""
            for cc in criteria:
                assert len(cc) > 0
                instructions += "- " + cc + "\n"
            question = f"{question_raw}\n\n{instructions.strip()}"
        elif q_type == "type5":
            instructions = ""
            for cc in criteria:
                assert len(cc) > 0
                if cc[-1] != "," and cc[-1] != "." and cc[-1] != "?" and cc[-1] != "!":
                    instructions += cc + ". "
                else:
                    instructions += cc + " "
            question = f"{instructions.strip()}\n\n{question_raw}"
        elif q_type == "type6":
            instructions = ""
            for cc in criteria:
                assert len(cc) > 0
                instructions += cc + "\n"
            question = f"{question_raw}\n\n{instructions.strip()}"

        return question
