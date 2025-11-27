class EdgeGraph:
    def __init__(self, hallucination_grader, code_evaluator):
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator

    def decide_to_generate(self, state):
        """
        Determines whether to generate an answer or re-generate a question based on the relevance of filtered documents to the input question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """
        print("---Entering retrieved document-question relevance judgment---")

        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---Decision: All retrieved documents are irrelevant to the question, transforming query---")
            return "transform_query"
        else:

            print("---Decision: Generate final response---")
            return "generate"

    def grade_generation_v_documents_and_question(self, state):
        """
        Evaluates the generated answer based on its grounding in the documents and its ability to answer the question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---Checking for model hallucination output---")
        question = state["input"]
        documents = state["documents"]
        generation = state["generation"]

        score = self.hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            print("---Decision: Generated content is based on established facts from retrieved documents---")

            print("---Checking if final response is relevant to input question---")
            score = self.code_evaluator.invoke({"input": question, "generation": generation, "documents": documents})
            grade = score["score"]
            if grade == "yes":
                print("---Judgment: Generated response is relevant to input question---")
                return "useful"
            else:
                print("---Judgment: Generated response is not relevant to input question---")
                return "not useful"
        else:
            print("---Judgment: Generated response is not relevant to retrieved documents, model is hallucinating---")
            return "not supported"
