import json
from typing import Optional, List, Dict

class SQLNode():
    def __init__(self, plain_content:List, plan_parameters:Dict,  plan_runtime: float) -> None:
        self.plain_content = plain_content
        self.plan_parameters = plan_parameters
        self.children : Optional[List['SQLNode']] = []
        self.plan_runtime = plan_runtime
        # self.join_conds = join_conds


class SQLTree():
    def __init__(self, parsed_sql:Dict) -> None:
        plain_content = parsed_sql["plain_content"]
        plan_parameters = parsed_sql["plan_parameters"]
        children = parsed_sql["children"]
        plan_runtime = parsed_sql["plan_runtime"]
        # join_conds = parsed_sql["join_conds"]
        self.root : Optional[SQLNode] = SQLNode(plain_content, plan_parameters, plan_runtime)
        self._insert_children_sql(self.root, children)
        
    def _insert_children_sql(self, node: SQLNode, children):
        for parsed_sql in children:
            plain_content = parsed_sql["plain_content"]
            plan_parameters = parsed_sql["plan_parameters"]
            children = parsed_sql["children"]
            plan_runtime = parsed_sql["plan_runtime"]
            # join_conds = parsed_sql["join_conds"]
            new_node = SQLNode(plain_content, plan_parameters, plan_runtime)
            node.children.append(new_node)
            self._insert_children_sql(new_node, children)
        
        return

class workload_dataloader():
    def __init__(self, filename) -> None:
        with open(filename, "r") as f:
            data = json.load(f)
        parsed_sqls = data["parsed_plans"]
        self.sql_forest = []
        for parsed_sql in parsed_sqls:
            tree = SQLTree(parsed_sql)
            self.sql_forest.append(tree)
    
    def get_data(self):
        return self.sql_forest

if __name__ == "__main__":
    filename = "../runs/deepdb_augmented/accidents/workload_100k_s1_c8220.json"
    workload = workload_dataloader(filename)
    tree = workload.get_data()[0]
    print(len(tree.root.children[0].children[0].children[0].children))
