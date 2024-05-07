import sys
import json
import re

def parse_5000_runtime(path):
  with open(path, "r") as f:
    augmented = json.load(f)

  runtime_parsed = []

  for i in range(len(augmented['parsed_plans'])):
    plan = augmented['parsed_plans'][i]
    runtime_parsed.append(plan['plan_runtime'])

  return runtime_parsed

def generate_plain(path, runtime_parsed):

  with open(path, "r") as f:
    raw_data = json.load(f)

  plain_statemnet_5000 = []
  plain_statement = []
  runtime_all = []
  runtime_5000 = []

  count = 0

  for i in range(len(raw_data['query_list'])):
    query = raw_data['query_list'][i]
    sql_text = query['sql']
    # print(i, "\n")
    # print(query['analyze_plans'])
    if not query['analyze_plans'] == []:
      runtime_text = query['analyze_plans'][0][-1][0]
      runtime = float(re.search(r'(\d+\.\d+)\s*ms', runtime_text).group(1))

      if runtime >= 100 and runtime <= 300000:
        data_entry = {"sql": sql_text, "runtime_ms": runtime}
        plain_statement.append(data_entry)
        runtime_all.append(runtime)

        if count < 5000 and runtime == runtime_parsed[count]:
          runtime_5000.append(runtime)
          plain_statemnet_5000.append(data_entry)
          count += 1

  # print(len(test_))
  with open('../plain_statement.json', 'w') as json_file:
      json.dump(plain_statement, json_file, indent=4)

  with open('../plain_statement_5000.json', 'w') as json_file:
      json.dump(plain_statemnet_5000, json_file, indent=4)

  return runtime_all, runtime_5000

if __name__ == "__main__":
  runtime_all, runtime_5000 = generate_plain(sys.argv[1], parse_5000_runtime(sys.argv[2]))