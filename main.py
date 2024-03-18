# https://platform.openai.com/docs/api-reference

from openai import OpenAI
import pandas as pd
import json
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
import matplotlib.pyplot as plt
from vertexai.preview.generative_models import GenerativeModel
from vertexai.language_models import ChatModel


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
    for row in data:  # --> TODO: Das '[:1]' beschränkt es auf die erste Zeile
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
        predictions.append({"model": "gpt-4-0125", "ProjectID": row["ProjectID"], "RequirementText": row["RequirementText"], "RequirementClass": row["RequirementClass"], "PredictionClass": response.choices[0].message.content})
    # Close JSON file
    f.close()
    # Save predictions to JSON file
    with open("requirements_predictions_gpt_4.json", "w") as fp:
        json.dump(predictions, fp)


def ask_gemini():
    # Open json file (filled with requirements)
    f = open("requirements.json")
    # Read the file and load them into a list of dicts
    data = json.load(f)
    # Need of array to save predictions
    predictions = []
    # Iterate through requirements
    for row in data:  # --> TODO: Das '[:1]' beschränkt es auf die erste Zeile
        print("RequirementText: " + row["RequirementText"])
        # Instance of the Gemini client in order to communicate with the current model
        model = GenerativeModel("gemini-1.0-pro-001")
        chat = model.start_chat()
        response = chat.send_message("Decide the class (choices: F, A, L, LF, MN, O, PE, SC, SE, US, FT, PO) of the following requirement specification. The answer only consists of the class: " + row["RequirementText"])
        # Ask Gemini to classify a single requirement specification
        print("Answer: " + response.text)
        # Save prediction row to array
        predictions.append({"model": "gemini-1.0-pro-001", "ProjectID": row["ProjectID"], "RequirementText": row["RequirementText"], "RequirementClass": row["RequirementClass"], "PredictionClass": response.text})
    # Close JSON file
    f.close()
    # Save predictions to JSON file
    with open("requirements_predictions_gemini-1.0-pro-001.json", "w") as fp:
        json.dump(predictions, fp)


def ask_palm():
    # Open json file (filled with requirements)
    f = open("requirements.json")
    # Read the file and load them into a list of dicts
    data = json.load(f)
    # Need of array to save predictions
    predictions = []
    # Iterate through requirements
    for row in data:  # --> TODO: Das '[:1]' beschränkt es auf die erste Zeile
        print("RequirementText: " + row["RequirementText"])
        # Instance of the PaLM2 client in order to communicate with the current model
        model = ChatModel.from_pretrained("chat-bison-32k@002")
        chat = model.start_chat()
        response = chat.send_message("Decide the class (choices: F, A, L, LF, MN, O, PE, SC, SE, US, FT, PO) of the following requirement specification. The answer only consists of the class: " + row["RequirementText"])
        # Ask Bison model to classify a single requirement specification
        print("Answer: " + response.text)
        # Save prediction row to array
        predictions.append({"model": "chat-bison-32k@002", "ProjectID": row["ProjectID"], "RequirementText": row["RequirementText"], "RequirementClass": row["RequirementClass"], "PredictionClass": response.text})
    # Close JSON file
    f.close()
    # Save predictions to JSON file
    with open("requirements_predictions_chat-bison-32k@002.json", "w") as fp:
        json.dump(predictions, fp)


def calculate_confusion_matrix(model_name):
    print(model_name)
    df = pd.read_json(f"requirements_predictions_{model_name}.json")
    # df["PredictionClassCorrected"] = df["PredictionClass"].apply(lambda x: "NewClass" if x not in ["F", "A", "L", "LF", "MN", "O", "PE", "SC", "SE", "US", "FT", "PO"] else x)
    y_true = df["RequirementClass"]
    y_pred = df["PredictionClass"]
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred)
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("fscore: {}".format(fscore))
    print("support: {}".format(support))
    matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=pd.concat([df["RequirementClass"], df["PredictionClass"]]).unique())
    disp.plot()
    plt.show()
    print("------")


if __name__ == "__main__":
    # format_requirements()
    # ask_gpt()
    # ask_gemini()
    # ask_palm()
    calculate_confusion_matrix("gpt_4")
    calculate_confusion_matrix("chat-bison-32k@002")
    calculate_confusion_matrix("gemini-1.0-pro-001")
