import os

from faker.providers import BaseProvider


class SeededDynamicProvider(BaseProvider):
    """A DynamicProvider that uses Faker's random instance instead of global random."""

    def __init__(self, generator, provider_name: str, elements: list):
        super().__init__(generator)
        self.provider_name = provider_name
        self.elements = elements

        # Dynamically create the provider method
        def provider_method(self):
            return self.random_element(self.elements)

        provider_method.__name__ = provider_name
        setattr(self.__class__, provider_name, provider_method)


PROVIDERS_DATA = "scripts/fake/"


def DynamicProviderFromFile(provider_name, file_path):
    """Create a Faker DynamicProvider from a file (.txt, .csv, .json).
    Returns a provider class that will use Faker's random instance.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    ext = os.path.splitext(file_path)[1].lower()

    # Load elements based on file type
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            elements = [line.strip() for line in f if line.strip()]
    elif ext == ".csv":
        import csv

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            elements = [row[0] for row in reader if row]  # take first column
    elif ext == ".json":
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            elements = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    class CustomProvider(SeededDynamicProvider):
        def __init__(self, generator):
            super().__init__(generator, provider_name, elements)

    return CustomProvider


subject = DynamicProviderFromFile("subject", PROVIDERS_DATA + "subject.txt")
location_phrase = DynamicProviderFromFile(
    "location_phrase", PROVIDERS_DATA + "location_phrase.txt"
)
verb = DynamicProviderFromFile("verb", PROVIDERS_DATA + "verb.txt")
profession = DynamicProviderFromFile("profession", PROVIDERS_DATA + "profession.txt")
item = DynamicProviderFromFile("item", PROVIDERS_DATA + "item.txt")
degree = DynamicProviderFromFile("degree", PROVIDERS_DATA + "degree.txt")
language = DynamicProviderFromFile("language", PROVIDERS_DATA + "language.txt")
fluency = DynamicProviderFromFile("fluency", PROVIDERS_DATA + "fluency.txt")
