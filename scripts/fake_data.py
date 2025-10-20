from datetime import datetime
from enum import StrEnum
from string import Formatter

from faker import Faker

fake = Faker()


def seed_instance(seed: int):
    """Seed the Faker instance for reproducibility."""
    fake.seed_instance(seed)


class TemplateVars(StrEnum):
    """Defines all possible variables for templates."""

    NAME = "NAME"
    BIRTH_DATE = "BIRTH_DATE"
    COUNTRY = "COUNTRY"
    ZODIAC_SIGN = "ZODIAC_SIGN"


def get_full_name() -> str:
    return f"{fake.unique.first_name()} {fake.unique.last_name()} {fake.unique.last_name()}"


def get_date_of_birth_str() -> str:
    return fake.unique.date_of_birth(minimum_age=10, maximum_age=60).strftime(
        "%d %B, %Y"
    )


def get_zodiac_sign(date_str):
    # Parse the input date string
    date = datetime.strptime(date_str, "%d %B, %Y")
    day = date.day
    month = date.month

    # Zodiac date ranges
    zodiac_signs = [
        ((1, 20), (2, 18), "Aquarius"),
        ((2, 19), (3, 20), "Pisces"),
        ((3, 21), (4, 19), "Aries"),
        ((4, 20), (5, 20), "Taurus"),
        ((5, 21), (6, 20), "Gemini"),
        ((6, 21), (7, 22), "Cancer"),
        ((7, 23), (8, 22), "Leo"),
        ((8, 23), (9, 22), "Virgo"),
        ((9, 23), (10, 22), "Libra"),
        ((10, 23), (11, 21), "Scorpio"),
        ((11, 22), (12, 21), "Sagittarius"),
        ((12, 22), (1, 19), "Capricorn"),
    ]

    # Determine the zodiac sign
    for start, end, sign in zodiac_signs:
        start_month, start_day = start
        end_month, end_day = end
        if (month == start_month and day >= start_day) or (
            month == end_month and day <= end_day
        ):
            return sign

    # Fallback (shouldn’t happen)
    return None


def get_zodiac_sign_from_data(data):
    date_str = data.get(TemplateVars.BIRTH_DATE.value)
    if not date_str:
        raise ValueError("Birth date must be generated before zodiac sign.")
    return get_zodiac_sign(date_str)


def get_country() -> str:
    return fake.unique.country()


GENERATOR_FUNCTIONS = {
    TemplateVars.NAME: get_full_name,
    TemplateVars.BIRTH_DATE: get_date_of_birth_str,
    TemplateVars.COUNTRY: get_country,
    TemplateVars.ZODIAC_SIGN: get_zodiac_sign_from_data,
}


def extract_keys(fmt: str):
    formatter = Formatter()
    return [field_name for _, field_name, _, _ in formatter.parse(fmt) if field_name]


def check_template_vars(template: str):
    keys = extract_keys(template)
    for key in keys:
        if key not in TemplateVars.__members__:
            raise ValueError(f"Invalid template variable: {key}")


def fill_template(template: str, test_cases: dict) -> str:
    keys = extract_keys(template)
    # print("Keys found in template:", keys)
    data_to_format = {}
    for key, func in GENERATOR_FUNCTIONS.items():
        if key in keys:
            try:
                data_to_format[key.value] = func(data_to_format)
            except TypeError:
                # backward compatibility for simple generators
                data_to_format[key.value] = func()
    # print("Data to format:", data_to_format)
    result = {}
    for key in test_cases:
        result[key] = []
        for case in test_cases[key]:
            prompt = case["prompt"].format(**data_to_format)
            completion = case["completion"].format(**data_to_format)
            result[key].append({"prompt": prompt, "completion": completion})
    return template.format(**data_to_format), result
