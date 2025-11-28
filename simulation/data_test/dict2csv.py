import pandas as pd
import pickle

# df = pd.read_csv("philly_vc.csv")
# d = dict(zip(df["vc"].values, df["node num"].values))

# with open(f"./vc_dict_homo.pkl", "wb") as f:
#     pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

# cluster_list = ["Venus", "Earth", "Saturn", "Uranus", "Philly"]

cluster_list = ["Venus"]

for i, v in enumerate(cluster_list):
    # vc_dict = pd.read_pickle(v + "/vc_dict_homo.pkl")
    vc_dict = pd.read_pickle(v + "/vc_dict.pkl")
    df = pd.DataFrame.from_dict(vc_dict, orient="index", columns=["num"])
    df.to_csv(v + "/vc_config.csv")

print(df.to_dict()["num"])
