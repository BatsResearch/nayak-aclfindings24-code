{
    "config_name": "privacy_qa",
    "dataset_name": "BatsResearch/bonito-experiment-eval",
    "templates": {
        "prompt_1": {
            "jinja": "Given the context, is this related to the question?\nContext: {{text}}\nQuestion: {{question}}|||{{answer}}",
            "answer_choices": "Relevant|||Irrelevant",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_2": {
            "jinja": 'Is this question\n"{{question}}"\nrelated to this context\n"{{text}}"?|||{% if answer == "Relevant" %} Yes {% else %} No {% endif %}',
            "answer_choices": "Yes|||No",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_3": {
            "jinja": 'Can this\n"{{text}}"\nhelp answer this question\n"{{question}}"?|||{% if answer == "Relevant" %} Yes {% else %} No {% endif %}',
            "answer_choices": "Yes|||No",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_4": {
            "jinja": 'As a lawyer, can you answer the question given the context?\nQuestion: {{question}}\nContext:{{text}}|||{% if answer == "Relevant" %} Yes {% else %} No {% endif %}',
            "answer_choices": "Yes|||No",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_5": {
            "jinja": 'Question:{{question}}\nContext:{{text}}\nIs the question related to the context?|||{% if answer == "Relevant" %} Yes {% else %} No {% endif %}',
            "answer_choices": "Yes|||No",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
    },
}
