import pandas as pd

def load_data(path):
  data_df = pd.read_csv(path)
  return data_df

def hello():
 print("Hello World!")


if __name__ == "__main__":
 hello()
 data_df = pd.read_csv("../../Data/IMDB Dataset.csv")
 print(data_df)

