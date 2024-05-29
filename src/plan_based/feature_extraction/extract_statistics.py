import json
from collections import deque

def extract_col_stat(parse_meta_path, col_meta_path):
  with open(parse_meta_path, "r") as f:
    parsed_meta = json.load(f)

  with open(col_meta_path, "r") as f:
    col_stat = json.load(f)

  col_stat_complete = dict()
  col_idx_map = dict()

  col_idx = 0

  for table in col_stat:
    cols = col_stat[table]
    for col in cols:
      col_dict = dict()
      col_name = table + "." + col
      col_idx_map[col_idx] = col_name
      
      col_idx += 1
      # print(cols[col])
      col_dict['nan_ratio'] = cols[col]['nan_ratio']
      col_dict['datatype'] = cols[col]['datatype']
      if cols[col]['datatype'] != 'misc':
        col_dict['num_unique'] = cols[col]['num_unique']
      col_stat_complete[col_name] = col_dict

  col_stat_supp = parsed_meta['database_stats']['column_stats']
  for col in col_stat_supp:
    col_name = col['tablename'] + "." + col['attname']
    col_stat_complete[col_name]['n_distinct'] = col['n_distinct']
    col_stat_complete[col_name]['avg_width'] = col['avg_width']
    col_stat_complete[col_name]['corr_with_row'] = col['correlation']
    col_stat_complete[col_name]['table_size'] = col['table_size']

  return col_idx_map, col_stat_complete

def extract_table_stat(parse_meta_path):
  with open(parse_meta_path, "r") as f:
    parsed_meta = json.load(f)

  table_stat_complete = dict()
  table_idx_map = dict()
  table_idx = 0

  for table_stat in parsed_meta['database_stats']['table_stats']:
    table_dict = dict()
    table_name = table_stat['relname']
    table_idx_map[table_idx] = table_name
    table_dict['reltuples'] = table_stat['reltuples']
    table_dict['relpages'] = table_stat['relpages']
    table_stat_complete[table_name] = table_dict
    table_idx += 1

  return table_idx_map, table_stat_complete

def extract_type_stat(parse_meta_path, col_idx_map, col_stat):

  col_type_idx_map = dict()
  for col in col_stat:
    col_type = col_stat[col]["datatype"]
    if col_type not in col_type_idx_map:
      col_type_idx_map[col_type] = len(col_type_idx_map)

  with open(parse_meta_path, "r") as f:
    parsed_meta = json.load(f)

  op_idx_map = dict()
  filter_op_idx_map = dict()

  for plan in parsed_meta['parsed_plans']:
    queue = deque([plan])

    while queue:
      level_count = len(queue)
      while level_count > 0:
        level_count -= 1
        node = queue.popleft()
        param = node['plan_parameters']

        op_name = param['op_name']
        if op_name not in op_idx_map:
          op_idx_map[op_name] = len(op_idx_map)
        
        if 'filter_columns' in param:
          filter_op = param['filter_columns']['operator']
          if filter_op == "AND":
            for filters in param['filter_columns']['children']:
              filter_op = filters['operator']
              if filter_op not in filter_op_idx_map:
                filter_op_idx_map[filter_op] = len(filter_op_idx_map)
          else:
            if filter_op not in filter_op_idx_map:
              filter_op_idx_map[filter_op] = len(filter_op_idx_map)

        for children in node['children']:
          queue.append(children)

  return col_type_idx_map, op_idx_map, filter_op_idx_map


if __name__ == '__main__':
  stat_complete = []

  col_idx_map, col_stat_complete = extract_col_stat("../../../datasets/plans/parsed/workload_5k_s1_c8220.json", "../../../datasets/stats/column_statistics.json")

  table_idx_map, table_stat_complete = extract_table_stat("../../../datasets/plans/parsed/workload_5k_s1_c8220.json")

  col_type_idx_map, op_idx_map, filter_op_idx_map = extract_type_stat("../../../datasets/plans/parsed/workload_5k_s1_c8220.json", col_idx_map, col_stat_complete)

  # print(col_type_idx_map, op_idx_map, filter_op_idx_map)

  stat_complete.append(col_idx_map)
  stat_complete.append(col_stat_complete)
  stat_complete.append(table_idx_map)
  stat_complete.append(table_stat_complete)
  stat_complete.append(col_type_idx_map)
  stat_complete.append(op_idx_map)
  stat_complete.append(filter_op_idx_map)

  stat_path = "../../../datasets/stats/stat_complete.json"
  with open(stat_path, "w") as f:
    json.dump(stat_complete, f)