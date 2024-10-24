from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

classes = ["Airport", "Bridge", "Center", "Desert", "Forest",
    "Industrial", "Mountain", "Pond", "Port", "Stadium"]

def setup_exp_folder():

    try:
        dir_folders = [int(x) for x in os.listdir("results")]
        new_folder_name = str(max(dir_folders) + 1)
        results_path = f"results/{new_folder_name}"
    except:
        results_path = f"results/0"

    if not os.path.exists(results_path):
        print (f"Results will be stored in: {results_path}")
        os.makedirs(results_path)

    return (results_path)

def plot_line(x, y, exp_name, title, location, legends):
    plt.plot(x, y)
    plt.legend(legends)
    plt.title(title)
    plt.savefig(f"{location + '/' + exp_name}.png")
    plt.close()

def create_class_report(targets, outputs):

    try:
        targets = targets.tolist()
        report = classification_report(outputs, targets, target_names=classes)
        df = pd.DataFrame(report).transpose()
        df.to_csv("class_report.csv", sep="|")

    except Exception as e:
        print ("Couldnt save the report")
        print (e)



 







