{
    "config_name": "vitaminc",
    "dataset_name": "BatsResearch/bonito-experiment-eval",
    "templates": {
        "prompt_1": {
            "jinja": 'Suppose {{evidence}} Can we infer that "{{claim}}"? yes, no or maybe?|||{% if label == "REFUTES" %} No {% elif label == "SUPPORTS" %} Yes {% else %} Maybe {% endif %}',
            "answer_choices": "No ||| Yes ||| Maybe",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_2": {
            "jinja": '{{evidence}} \n\nQuestion: Does this imply that "{{claim}}"? yes, no or maybe?|||{% if label == "REFUTES" %} No {% elif label == "SUPPORTS" %} Yes {% else %} Maybe {% endif %}',
            "answer_choices": "No ||| Yes ||| Maybe",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_3": {
            "jinja": 'Take the following as truth: {{evidence}} Then the following statement: "{{claim}}" is {{"true"}}, {{"false"}}, or {{"inconclusive"}}?|||{% if label == "REFUTES" %} False {% elif label == "SUPPORTS" %} True {% else %} Inconclusive {% endif %}',
            "answer_choices": "False ||| True ||| Inconclusive",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_4": {
            "jinja": '{{evidence}}\nBased on that information, is the claim: "{{claim}}" {{"true"}}, {{"false"}}, or {{"inconclusive"}}? |||{% if label == "REFUTES" %} False {% elif label == "SUPPORTS" %} True {% else %} Inconclusive {% endif %}',
            "answer_choices": "False ||| True ||| Inconclusive",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_5": {
            "jinja": '{{evidence}} Based on the previous passage, is it true that "{{claim}}"? Yes, no, or maybe? |||{% if label == "REFUTES" %} No {% elif label == "SUPPORTS" %} Yes {% else %} Maybe {% endif %}',
            "answer_choices": "No ||| Yes ||| Maybe",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
    },
}
