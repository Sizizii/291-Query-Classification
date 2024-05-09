import json

# Exploring the raw dataset.
file_name = "../runs/deepdb_augmented/accidents/workload_100k_s1_c8220.json"
with open(file_name, "r") as f:
    raw_accidents_normal_data = json.load(f)
print(raw_accidents_normal_data["parsed_plans"][0].keys())
print(len(raw_accidents_normal_data["parsed_plans"][0]["children"][0]["children"][0]["children"][0]["children"]))
# print(raw_accidents_normal_data["parsed_plans"][0]["children"])
