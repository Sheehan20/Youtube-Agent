#!/usr/bin/env python
# -*- coding: utf-8 -*-
# YouTube Agent Graph Nodes Module

from bili_server.generate_chain import create_generate_chain


class GraphNodes:
    def __init__(self, llm, retriever, retrieval_grader, hallucination_grader, code_evaluator, question_rewriter):
        self.llm = llm
        self.retriever = retriever
        self.retrieval_grader = retrieval_grader
        self.hallucination_grader = hallucination_grader
        self.code_evaluator = code_evaluator
        self.question_rewriter = question_rewriter
        self.generate_chain = create_generate_chain(llm)

    async def retrieve(self, state):
        """
        Retrieve documents based on the input question and add them to the graph state.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---Node: Start Retrieval---")
        question = state["input"]

        # Execute retrieval
        documents = await self.retriever.get_retriever(keywords=[question], page=1)
        print(f"Retrieved Docs: {documents}")
        return {"documents": documents, "input": question}

    def generate(self, state):
        """
        Generate answer using the input question and retrieved documents, and add the generation to the graph state.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---Node: Generate Response---")

        question = state["input"]
        documents = state["documents"]

        # Generate based on RAG
        generation = self.generate_chain.invoke({"context": documents, "input": question})
        print(f"Generated response: {generation}")
        return {"documents": documents, "input": question, "generation": generation}

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """
        print("---Node: Check if retrieved documents are relevant to the question---")
        question = state["input"]
        documents = state["documents"]


        filtered_docs = []

        for d in documents:
            score = self.retrieval_grader.invoke({"input": question, "document": d.page_content})
            grade = score["score"]
            if grade == "yes":
                print("---Evaluation result: Retrieved document is relevant to question---")
                filtered_docs.append(d)
            else:
                print("---Evaluation result: Retrieved document is not relevant to question---")
                continue

        return {"documents": filtered_docs, "input": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """
        print("---Node: Rewrite user input question---")

        question = state["input"]
        documents = state["documents"]

        # Question rewrite
        better_question = self.question_rewriter.invoke({"input": question})
        print(f"Rewritten question: {better_question}")
        return {"documents": documents, "input": better_question}
