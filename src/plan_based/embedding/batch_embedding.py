# from src.plan_based.embedding.plan_embedding import *
from plan_embedding import *

def _batch_merge_vec(batch_vec, plan_vec):
  # if plan has larger number of levels, create a new level list and append
  for level, level_vec in enumerate(plan_vec):
    if level >= len(batch_vec):
      batch_vec.append([])
    batch_vec[level] += level_vec

  return batch_vec

def _batch_merge_mapping(batch_map, plan_map):
  for level, level_map in enumerate(plan_map):
    if level >= len(batch_map):
      batch_map.append([])

    # to find if there are preceding parent nodes on the same level
    # if found, should increments its children node mapping (on next level)
    if level < len(batch_map) - 1:
      batch_level_base = len(batch_map[level + 1])
      for i in range(len(level_map)):
        if level_map[i][0] > 0:
          level_map[i][0] += batch_level_base
        if level_map[i][1] > 0:
          level_map[i][1] += batch_level_base
      
    batch_map[level] += level_map

  return batch_map

def pad_to_max_len(batch_vec, max_num):
  return np.array([np.pad(vec, ((0, max_num - len(vec)), (0, 0)), 'constant') for vec in batch_vec])


def batch_embed(plan_trees, max_filter, num_filter_attr, op_stats, col_id_map, col_stats, table_id_map, table_stats, type_id_map, filter_op_id_map, num_attr):

  target_runtime = []
  batch_op, batch_attr, batch_filter, batch_output, batch_mapping = [], [], [], [], []

  for plan_tree in plan_trees:
    plan_root = plan_tree.root
    target_runtime.append(plan_root.plan_runtime)

    plan_op, plan_attr, plan_filter, plan_output, plan_mapping = plan_embed(plan_root, max_filter, num_filter_attr, op_stats, col_id_map, col_stats, table_id_map, table_stats, type_id_map, filter_op_id_map, num_attr)

    batch_op = _batch_merge_vec(batch_op, plan_op)
    batch_attr = _batch_merge_vec(batch_attr, plan_attr)
    batch_filter = _batch_merge_vec(batch_filter, plan_filter)
    batch_output = _batch_merge_vec(batch_output, plan_output)

    batch_mapping = _batch_merge_mapping(batch_mapping, plan_mapping)


  max_num_node_per_level = 0
  for op in batch_op:
    max_num_node_per_level = max(max_num_node_per_level, len(op))
  
  batch_op_pad = pad_to_max_len(batch_op, max_num_node_per_level)
  batch_attr_pad = pad_to_max_len(batch_attr, max_num_node_per_level)
  batch_filter_pad = pad_to_max_len(batch_filter, max_num_node_per_level)
  batch_output_pad = pad_to_max_len(batch_output, max_num_node_per_level)
  batch_mapping_pad = pad_to_max_len(batch_mapping, max_num_node_per_level)

  return target_runtime, batch_op_pad, batch_attr_pad, batch_filter_pad, batch_output_pad, batch_mapping_pad, batch_op, batch_attr, batch_filter, batch_output, batch_mapping


