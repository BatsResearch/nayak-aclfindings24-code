from typing import Union

import os
import pkg_resources
from promptsource.templates import Template

# Local path to the folder containing the templates (credit: promptsource)
TEMPLATES_FOLDER_PATH = pkg_resources.resource_filename(__name__, "templates")


def template_collector(path):
    """
    A decorator that collects all Python files with dictionary data in a path
    and returns a dictionary.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            dict_data = {}
            for filename in os.listdir(path):
                if filename.endswith(".py"):
                    filepath = os.path.join(path, filename)
                    filename = filename.replace(".py", "")
                    with open(filepath, "r") as f:
                        file_contents = f.read()
                        file_dict = eval(file_contents)
                        dataset_name = file_dict["dataset_name"]
                        config_name = file_dict["config_name"]
                        dict_data[(dataset_name, config_name)] = file_dict["templates"]

            return func(dict_data, *args, **kwargs)

        return wrapper

    return decorator


@template_collector(TEMPLATES_FOLDER_PATH)
def gather_templates(templates_dict: dict) -> dict:
    """The function collectrs all the templates in the templates directory.
    If you want to choose only one template, use the choose_template function.
    """
    return templates_dict


def choose_template(
    dataset: str, config_name: str = None, template_name: str = None
) -> Template:
    """The function chooses template for the given dataset and the template name.

    Args:
        dataset (str): name of the dataset from huggingface
            (eg. scitail or bigbio/mednli).
        template_name (str): name of the template from the list of
            templates for a dataset.

    Returns:
        Template: the template object which contains the jinja template,
            answer choices, and the reference.
    """
    templates_dict = gather_templates()
    template = templates_dict[(dataset, config_name)][template_name]

    return Template(
        name=template_name,
        jinja=template["jinja"],
        reference=template["reference"],
        answer_choices=template["answer_choices"],
    )
