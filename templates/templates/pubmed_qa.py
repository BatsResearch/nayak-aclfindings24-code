{
    "config_name": "pubmed_qa",
    "dataset_name": "BatsResearch/bonito-experiment-eval",
    "templates": {
        "Given a passage (question at end)": {  # name of the template
            "jinja": 'Given a passage: {{ context.contexts | join(" ") }}\n\nAnswer the question: {{question}}\n\nSummarize the above answer as YES, NO, or MAYBE? |||\n{{final_decision}}',  # jinja template
            "answer_choices": "yes ||| no ||| maybe",  # answer choices separated by |||
            "reference": "Similar to BioASQ from https://arxiv.org/pdf/2206.15076.pdf",  # source of the prompt style if you found from a website.
        },
        "Im a doctor": {  # name of the template
            "jinja": 'I\'m a doctor and I want to answer the question "{{question}}" using The following passage:\n\n{{ context.contexts | join(" ") }}\n\nSummarize the above answer as YES, NO, or MAYBE? |||\n{{final_decision}}',  # jinja template
            "answer_choices": "yes ||| no ||| maybe",  # answer choices separated by |||
            "reference": "Similar to BioASQ from https://arxiv.org/pdf/2206.15076.pdf",  # source of the prompt style if you found from a website.
        },
        "What is the answer": {  # name of the template
            "jinja": 'What is the answer to the question "{{question}}" based on The following passage:\n\n{{ context.contexts | join(" ") }}\n\nSummarize the above answer as YES, NO, or MAYBE?|||\n{{final_decision}}',  # jinja template
            "answer_choices": "yes ||| no ||| maybe",  # answer choices separated by |||
            "reference": "Similar to BioASQ from https://arxiv.org/pdf/2206.15076.pdf",  # source of the prompt style if you found from a website.
        },
        "Please answer": {  # name of the template
            "jinja": 'Please answer the question "{{question}}" using The following passage:\n\n{{ context.contexts | join(" ") }}\n\nSummarize the above answer as YES, NO, or MAYBE?|||\n{{final_decision}}',  # jinja template
            "answer_choices": "yes ||| no ||| maybe",  # answer choices separated by |||
            "reference": "Similar to BioASQ from https://arxiv.org/pdf/2206.15076.pdf",  # source of the prompt style if you found from a website.
        },
        "Given a passage (question at start)": {  # name of the template
            "jinja": 'Given the following passage, answer the question: "{{question}}"\n\nPassage: {{ context.contexts | join(" ") }}\n\nSummarize the above answer as YES, NO, or MAYBE?|||\n{{final_decision}}',  # jinja template
            "answer_choices": "yes ||| no ||| maybe",  # answer choices separated by |||
            "reference": "Similar to BioASQ from https://arxiv.org/pdf/2206.15076.pdf",  # source of the prompt style if you found from a website.
        },
    },
}
