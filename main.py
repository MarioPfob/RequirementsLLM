# https://platform.openai.com/docs/api-reference

from openai import OpenAI
import pandas as pd


def main():
    # read requirements from csv
    df = pd.read_csv("requirements.csv", sep=";")
    # df["RequirementText"] = df["RequirementText"].str.strip()  # --> sind die Whitespaces Absicht?
    print(df.describe)
    df.to_json("requirements.json", orient="records")


if __name__ == "__main__":
    main()