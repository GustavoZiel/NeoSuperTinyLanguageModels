"""This module provides functions to generate fake data using the Faker library
and custom providers defined in `custom_providers.py`.
It includes functions to generate names, dates, locations, and other entities,
as well as a template filling mechanism.
"""

import os
import sys
from datetime import datetime
from enum import StrEnum
from string import Formatter

# Add current directory to sys.path to allow imports from custom_providers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_providers import (
    color,
    degree,
    fluency,
    item,
    language,
    location_phrase,
    profession,
    subject,
    verb,
)
from faker import Faker

all_providers = [
    degree,
    fluency,
    item,
    language,
    location_phrase,
    profession,
    subject,
    verb,
    color,
]

fake = Faker()
for provider in all_providers:
    fake.add_provider(provider)


def seed(seed: int):
    """Seed the Faker instance for reproducibility."""
    fake.seed_instance(seed)


class TemplateVars(StrEnum):
    """Defines all possible variables for templates."""

    NAME = "NAME"
    BIRTH_DATE = "BIRTH_DATE"
    COUNTRY = "COUNTRY"
    CITY = "CITY"
    COLOR = "COLOR"
    SSN = "SSN"
    SUBJECT = "SUBJECT"
    LOCATION = "LOCATION"
    VERB = "VERB"
    PROFESSION = "PROFESSION"
    COMPANY = "COMPANY"
    LOCATION_PHRASE = "LOCATION_PHRASE"
    DEGREE = "DEGREE"
    FLUENCY = "FLUENCY"
    ITEM = "ITEM"
    LANGUAGE = "LANGUAGE"
    TEST = "TEST"


def get_full_name() -> str:
    """Generate a unique full name."""
    # return fake.unique.name()
    return f"{fake.unique.first_name()} {fake.unique.last_name()} {fake.unique.last_name()}"


def get_date_of_birth_str() -> str:
    """Generate a unique date of birth string."""
    return fake.unique.date_of_birth(minimum_age=10, maximum_age=60).strftime(
        "%B %d, %Y"
    )


def get_country() -> str:
    """Generate a unique country name."""
    return fake.unique.country()


def get_city() -> str:
    """Generate a unique city name."""
    return fake.unique.city()


def get_color() -> str:
    """Generate a unique color name."""
    return fake.unique.color()


def get_ssn() -> str:
    """Generate a unique SSN."""
    return fake.unique.ssn()


def get_subject() -> str:
    """Generate a unique subject."""
    return fake.unique.subject()


def get_location() -> str:
    """Generate a unique location."""
    return fake.unique.location()


def get_verb() -> str:
    """Generate a unique verb."""
    return fake.unique.verb()


def get_profession() -> str:
    return fake.unique.profession()


def get_company() -> str:
    return fake.unique.company()


def get_location_phrase() -> str:
    return fake.unique.location_phrase()


def get_degree() -> str:
    return fake.unique.degree()


def get_fluency() -> str:
    return fake.fluency()


def get_item() -> str:
    return fake.unique.item()


def get_language() -> str:
    return fake.unique.language()


GENERATOR_FUNCTIONS = {
    TemplateVars.NAME: get_full_name,
    TemplateVars.BIRTH_DATE: get_date_of_birth_str,
    TemplateVars.COUNTRY: get_country,
    TemplateVars.CITY: get_city,
    TemplateVars.COLOR: get_color,
    TemplateVars.SSN: get_ssn,
    TemplateVars.SUBJECT: get_subject,
    TemplateVars.LOCATION: get_location,
    TemplateVars.VERB: get_verb,
    TemplateVars.PROFESSION: get_profession,
    TemplateVars.COMPANY: get_company,
    TemplateVars.LOCATION_PHRASE: get_location_phrase,
    TemplateVars.DEGREE: get_degree,
    TemplateVars.FLUENCY: get_fluency,
    TemplateVars.ITEM: get_item,
    TemplateVars.LANGUAGE: get_language,
    TemplateVars.TEST: lambda: "test",
}


def extract_keys(fmt: str):
    formatter = Formatter()
    keys_with_pos = {}
    i = 0
    for _, field_name, _, _ in formatter.parse(fmt):
        if field_name and field_name not in keys_with_pos:
            keys_with_pos[field_name] = i
            i += 1
    return keys_with_pos


def check_template_vars(template: str):
    keys = extract_keys(template)
    for key in keys:
        if key not in TemplateVars.__members__:
            raise ValueError(f"Invalid template variable: {key}")


def capitalize_first(s):
    return s[0].upper() + s[1:] if s else s


def fill_template(template: str, test_cases: dict) -> str:
    # fake.unique.clear()
    keys_with_pos = extract_keys(template)
    keys = list(keys_with_pos.keys())
    data_to_format = {}
    for key, func in GENERATOR_FUNCTIONS.items():
        if key in keys:
            try:
                value = func(data_to_format)
            except TypeError:
                value = func()

            # if keys_with_pos.get(key.value) == 0 and isinstance(value, str):
            #     value = capitalize_first(value)

            data_to_format[key.value] = value

    result = {}
    for key in test_cases:
        result[key] = []
        for case in test_cases[key]:
            # prompt = case["prompt"].format(**data_to_format)
            prompt = capitalize_first(case["prompt"].format(**data_to_format))
            completion = case["completion"].format(**data_to_format)
            # completion = capitalize_first(case["completion"].format(**data_to_format))
            result[key].append({"sentence": prompt, "answer": completion})
    # print("Data to format:", data_to_format)
    # print("Formatted template:", template.format(**data_to_format))
    # print("Generated test cases:", result)
    return capitalize_first(template.format(**data_to_format)), result
