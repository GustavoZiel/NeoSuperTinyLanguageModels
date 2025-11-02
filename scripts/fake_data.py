from datetime import datetime
from enum import StrEnum
from string import Formatter

from faker import Faker
from providers import (
    degree,
    fluency,
    item,
    language,
    location_phrase,
    profession,
    subject,
    verb,
)

all_providers = [
    degree,
    fluency,
    item,
    language,
    location_phrase,
    profession,
    subject,
    verb,
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


def get_full_name() -> str:
    # return fake.unique.name()
    return f"{fake.unique.first_name()} {fake.unique.last_name()} {fake.unique.last_name()}"


def get_date_of_birth_str() -> str:
    return fake.unique.date_of_birth(minimum_age=10, maximum_age=60).strftime(
        "%B %d, %Y"
    )


def get_country() -> str:
    return fake.unique.country()


def get_city() -> str:
    return fake.unique.city()


def get_color() -> str:
    return fake.unique.color_name()


def get_ssn() -> str:
    return fake.unique.ssn()


def get_subject() -> str:
    return fake.unique.subject()


def get_location() -> str:
    return fake.unique.location()


def get_verb() -> str:
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

            if keys_with_pos.get(key.value) == 0 and isinstance(value, str):
                value = capitalize_first(value)

            data_to_format[key.value] = value

    result = {}
    for key in test_cases:
        result[key] = []
        for case in test_cases[key]:
            prompt = capitalize_first(case["prompt"].format(**data_to_format))
            completion = capitalize_first(case["completion"].format(**data_to_format))
            result[key].append({"prompt": prompt, "completion": completion})
    return template.format(**data_to_format), result
