import numpy as np
import re

def node_embed(node_param, max_filter, num_filter_attr, op_stats, col_id_map, col_stats, table_id_map, table_stats, type_id_map, filter_op_id_map, num_attr):

  # operations
  op_vec = np.array([0 for _ in range(len(op_stats))])
  op_vec[op_stats[node_param['op_name']]] = 1

  # attributes
  attr_vec = np.array([0.0 for _ in range(num_attr)])
  attr_vec[0] = node_param['est_card']
  attr_vec[1] = node_param['est_width']
  if 'table' in node_param:
    table_name = table_id_map[str(node_param['table'])]
    table_stat = table_stats[table_name]
    attr_vec[2] = table_stat['reltuples']
    attr_vec[3] = table_stat['relpages']


  # predicates
  # col filter_op literal
  filter_vec = np.array([0 for _ in range(max_filter * num_filter_attr)])
  if 'filter_columns' in node_param:
    filter_idx = 0
    num_data_type = len(type_id_map)
    # 5 col statistics are used
    num_col_fileds = num_data_type + 5
    num_filter_op = len(filter_op_id_map)

    filters = node_param['filter_columns']
    if filters['operator'] == "AND":
      filters = node_param['filter_columns']['children']

    for fil in filters:
      # vector includes:
      # column related:
      #   col data type, num_unique, nan_ratio, attr correlation with row number, avg width, table_size
      # filter_op
      # literal_related: unique values of literal
      fil_start = filter_idx * num_filter_attr
      col_stat = col_stats[col_id_map[str(fil['column'])]]
      col_data_type = col_stat["datatype"]
      filter_vec[fil_start + type_id_map[col_data_type]] = 1
      filter_vec[fil_start + num_data_type] = col_stat["num_unique"]
      filter_vec[fil_start + num_data_type + 1] = col_stat["nan_ratio"]
      filter_vec[fil_start + num_data_type + 2] = col_stat["corr_with_row"]
      filter_vec[fil_start + num_data_type + 3] = col_stat["avg_width"]
      filter_vec[fil_start + num_data_type + 4] = col_stat["table_size"]

      filter_op_idx = filter_op_id_map[fil['operator']]
      filter_vec[fil_start + num_col_fileds + filter_op_idx] = 1

      literal = fil['literal']
      filter_vec[fil_start + num_col_fileds + num_filter_op] = 1
      if isinstance(literal, str):
        l_col = literal.strip()
        if re.match(r'^\w+\.\w+$', l_col):
          filter_vec[fil_start + num_col_fileds + num_filter_op] = col_stats[l_col]["num_unique"]

      filter_idx += 1

  # outputs
  # output aggregation type inclue COUNT, AVG, SUM, and NONE
  num_col = len(col_id_map)
  output_vec = np.array([0 for _ in range(num_col * 3 + 1)])
  output_avg_start = 1
  output_sum_start = num_col + 1
  output_none_start = 2*num_col + 1
  if 'output_columns' in node_param:
    for output in node_param['output_columns']:
      agg_op = output['aggregation']

      if agg_op == 'COUNT':
        output_vec[0] = 1
      elif agg_op == 'AVG':
        for output_col in output['columns']:
          output_vec[output_avg_start + output_col] = 1
      elif agg_op == "SUM":
        for output_col in output['columns']:
          output_vec[output_sum_start + output_col] = 1
      else:
        for output_col in output['columns']:
          output_vec[output_none_start + output_col] = 1

  return op_vec, attr_vec, filter_vec, output_vec