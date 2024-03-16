# https://platform.openai.com/docs/api-reference

from openai import OpenAI
import pandas as pd
import json
import os


def format_requirements():
    # Requirements will be loaded from CSV file into a pandas dataframe 
    df = pd.read_csv("requirements.csv", sep=";")
    # df["RequirementText"] = df["RequirementText"].str.strip()  # --> TODO: Sind die Whitespaces Absicht?
    # See if columns etc. look fine
    print(df.dtypes)
    # Write dataframe to JSON structure
    df.to_json("requirements.json", orient="records")


def ask_gpt():
    # Open json file (filled with requirements)
    f = open("requirements.json")
    # Read the file and load them into a list of dicts
    data = json.load(f)
    # Need of array to save predictions
    predictions = []
    # Iterate through requirements
    for row in data[:1]:  # --> TODO: Das ':1' beschränkt es auf die erste Zeile
        print("RequirementText: " + row["RequirementText"])
        # Instance of the OpenAI client in order to communicate with GPT
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))  # --> TODO: Der Key ist in einer .env Datei (nicht im Repo)
        # Ask GPT to classify a single requirement specification
        response = client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": "Decide the class (choices: F, A, L, LF, MN, O, PE, SC, SE, US, FT, PO) of the following requirement specification. The answer only consists of the class: " + row["RequirementText"]},
            ]
        )
        print("Answer: " + response.choices[0].message.content)
        # Save prediction row to array
        predictions.append({"ProjectID": row["ProjectID"], "RequirementText": row["RequirementText"], "RequirementClass": row["RequirementClass"], "PredictionClass": response.choices[0].message.content})
    # Close JSON file
    f.close()
    # Save predictions to JSON file
    with open("requirements_predictions.json", "w") as fp:
        json.dump(predictions, fp)


if __name__ == "__main__":
    format_requirements()
    ask_gpt()
