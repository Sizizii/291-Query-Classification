# from src.plan_based.embedding.node_embedding import *
from node_embedding import *
from collections import deque

def plan_embed(root_node, max_filter, num_filter_attr, op_stats, col_id_map, col_stats, table_id_map, table_stats, type_id_map, filter_op_id_map, num_attr):

  node_queue = deque([root_node])

  plan_op, plan_attr, plan_filter, plan_output, plan_mapping = [], [], [], [], []

  while node_queue:
    level_count = len(node_queue)

    plan_op.append([])
    plan_attr.append([])
    plan_filter.append([])
    plan_output.append([])
    plan_mapping.append([])

    children_sum = 0

    while level_count > 0:
      level_count -= 1
      node = node_queue.popleft()
      node_param = node.plan_parameters
      op_vec, attr_vec, filter_vec, output_vec = node_embed(node_param, max_filter, num_filter_attr, op_stats, col_id_map, col_stats, table_id_map, table_stats, type_id_map, filter_op_id_map, num_attr)

      plan_op[-1].append(op_vec)
      plan_attr[-1].append(attr_vec)
      plan_filter[-1].append(filter_vec)
      plan_output[-1].append(output_vec)

      if len(node.children) == 2:
        plan_mapping[-1].append([child.idx + children_sum for child in node.children])
      elif len(node.children) == 1:
        plan_mapping[-1].append([node.children[0].idx + children_sum , 0])
      else:
        plan_mapping[-1].append([0,0])

      children_sum += len(node.children)

      for child in node.children:
        node_queue.append(child)

  return plan_op, plan_attr, plan_filter, plan_output, plan_mapping