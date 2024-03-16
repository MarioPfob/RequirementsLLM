# https://platform.openai.com/docs/api-reference

from openai import OpenAI
import pandas as pd
import json
import os


def format_requirements():
    # read requirements from csv
    df = pd.read_csv("requirements.csv", sep=";")
    # df["RequirementText"] = df["RequirementText"].str.strip()  # --> sind die Whitespaces Absicht?
    print(df.describe)
    df.to_json("requirements.json", orient="records")


def ask_gpt():
    f = open("requirements.json")
    data = json.load(f)
    for row in data[:1]:
        print("RequirementText: " + row["RequirementText"])
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
        model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": "Decide the class (choices: F, A, L, LF, MN, O, PE, SC, SE, US, FT, PO) of the following requirement specification: " + row["RequirementText"]},
            ]
        )
        print("Answer: " + response.choices[0].message.content)
    f.close()


if __name__ == "__main__":
    format_requirements()
    ask_gpt()