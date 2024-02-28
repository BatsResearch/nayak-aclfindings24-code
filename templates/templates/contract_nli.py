{
    "config_name": "contract_nli",
    "dataset_name": "BatsResearch/bonito-experiment-eval",
    "templates": {
        "prompt_1": {
            "jinja": 'Suppose {{premise}} Can we infer that "{{hypothesis}}"? yes, no or maybe?|||{{answer_choices[label]}}',
            "answer_choices": "No ||| Yes ||| Maybe",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_2": {
            "jinja": '{{premise}} \n\nQuestion: Does this imply that "{{hypothesis}}"? yes, no or maybe?|||{{answer_choices[label]}}',
            "answer_choices": "No ||| Yes ||| Maybe",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_3": {
            "jinja": 'Take the following as truth: {{premise}} Then the following statement: "{{hypothesis}}" is {{"true"}}, {{"false"}}, or {{"inconclusive"}}?|||{{answer_choices[label]}}',
            "answer_choices": "False ||| True ||| Inconclusive",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_4": {
            "jinja": '{{premise}} Based on that information, is the claim: "{{hypothesis}}" {{"true"}}, {{"false"}}, or {{"inconclusive"}}? |||{{ answer_choices[label]}}',
            "answer_choices": "False ||| True ||| Inconclusive",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
        "prompt_5": {
            "jinja": '{{premise}} Based on the previous passage, is it true that "{{hypothesis}}"? Yes, no, or maybe? ||| {{ answer_choices[label] }}',
            "answer_choices": "No ||| Yes ||| Maybe",  # answer choices separated by |||
            "reference": "",  # source of the prompt style if you found from a website.
        },
    },
}
