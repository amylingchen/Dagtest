from collections import deque
from z3 import *
from typing import Dict, List, Union
class DAGNode:
    """
    Represents a node in a Directed Acyclic Graph (DAG).

    Attributes:
        name (str): The unique name of the node.
        node_type (str): The type of the node ('INPUT', 'AND', 'OR', 'NOT', 'OUTPUT', etc.).
        children (list): List of child nodes connected to this node.
        value (int or None): The computed value of the node during evaluation.
    """
    def __init__(self,name: str, node_type: str, delay: int = 0):
        self.name = name
        self.node_type = node_type
        self.delay = delay
        self.children = []
        self.value = None
        self.values = []


class DAG:
    """
    Represents a Directed Acyclic Graph (DAG) for logical operations.

    Attributes:
        nodes (dict): A dictionary mapping node names to DAGNode objects.
        sorted_nodes (list): List of node names in topologically sorted order.
        adj (dict or None): Cached adjacency list representation of the DAG.
    """
    def __init__(self, adj=None):
        self.nodes = {}
        self.sorted_nodes = []
        self.adj = None
        self._critical_path = 0
        if adj is not None:
            self.create_dag(adj)

    def add_node(self, name: str, node_type: str, delay: int = 0):
        """
        Adds a node to the DAG.

        Args:
            name (str): The name of the node.
            node_type (str): The type of the node ('INPUT', 'AND', 'OR', 'NOT', 'OUTPUT', etc.).

        Raises:
            ValueError: If the node already exists.
        """
        if name in self.nodes:
            raise ValueError(f"Node {name} already exists.")
        self.nodes[name] = DAGNode(name, node_type,delay=delay)
        self.sorted_nodes = []
        self.adj = None

    def add_edge(self, parent_name, child_name):
        """
        Adds a directed edge from a parent node to a child node.

        Args:
            parent_name (str): The name of the parent node.
            child_name (str): The name of the child node.

        Raises:
            ValueError: If either the parent or child node does not exist.
        """
        parent = self.nodes.get(parent_name)
        child = self.nodes.get(child_name)
        if not parent or not child:
            raise ValueError(f"Parent {parent_name} or child {child_name} not found.")
        parent.children.append(child)
        self.sorted_nodes = []
        self.adj = None

    def create_dag(self, adj):
        """
        Constructs the DAG from an adjacency list representation.

        Args:
            adj (dict): A dictionary representing the DAG structure.

        Raises:
            ValueError: If there are inconsistencies in node types or missing parent nodes.
        """
        self.adj = adj
        for node, value in adj.items():
            node_type = 'INPUT' if value is None else value[0]
            delay =value[-1] if value and  len(value) == 3 else 0
            if node in self.nodes:
                if self.nodes[node].node_type != node_type:
                    raise ValueError(f"Node {node} type mismatch.")
            else:
                self.add_node(node, node_type,delay)

        for node, value in adj.items():
            if value is not None and len(value) >1:
                for parent in value[1]:
                    if parent not in self.nodes:
                        raise ValueError(f"Parent {parent} for {node} not found.")
                    self.add_edge(parent, node)

    def get_node_parents(self, node_name):
        """
        Returns the parent nodes of a given node.

        Args:
            node_name (str): The name of the node.

        Returns:
            list: A list of parent node names.

        Raises:
            ValueError: If the node does not exist.
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist.")
        return [parent.name for parent in self.nodes.values() if
                any(child.name == node_name for child in parent.children)]

    def get_adj(self):
        """
        Returns the adjacency list representation of the DAG.

        Returns:
            dict: A dictionary representing the DAG.
        """
        if not self.adj:
            adj = {}
            for node_name, node in self.nodes.items():
                if node.node_type == "INPUT":
                    adj[node_name] = ('INPUT',)
                else:
                    parents = self.get_node_parents(node_name)
                    if parents:
                        adj[node_name] = (node.node_type, *parents)
            self.adj = adj  # Cache the generated adjacency list
        return self.adj

    def topological_sort(self):
        """
        Performs topological sorting on the DAG.

        Returns:
            list: A list of node names in topologically sorted order.

        Raises:
            ValueError: If a cycle is detected in the graph.
        """
        in_degree = {name: 0 for name in self.nodes}
        max_delay = {name: 0 for name in self.nodes}


        for node in self.nodes.values():
            for child in node.children:
                in_degree[child.name] += 1
                max_delay[child.name] = max(max_delay[child.name],
                                            node.delay + max_delay.get(node.name, 0))


        self._critical_path = max(max_delay.values(), default=0)


        queue = deque([n for n in self.nodes.values() if in_degree[n.name] == 0])
        sorted_nodes = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for child in node.children:
                in_degree[child.name] -= 1
                if in_degree[child.name] == 0:
                    queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in DAG")

        self.sorted_nodes = sorted_nodes
        return sorted_nodes

    def critical_path_delay(self) -> int:
        """get delay of critical path"""
        if not self.sorted_nodes:
            self.topological_sort()
        return self._critical_path

    def evaluate(self, inputs, adj=None, outall=False):
        """
        Evaluates the DAG based on logical operations.

        Args:
            inputs (dict): A dictionary mapping input node names to their values.
            adj (dict, optional): The adjacency list representation of the DAG.
            outall (bool, optional): Whether to return all non-input node values. Defaults to False.

        Returns:
            dict: A dictionary mapping output node names to their computed values.

        Raises:
            ValueError: If required input values are missing or an unknown operation is encountered.
        """
        adj = adj or self.get_adj()
        input_nodes =  [name for name, node in self.nodes.items() if node.node_type == 'INPUT']
        # input_nodes = self.get_all_inputs()
        missing = [n for n in input_nodes if n not in inputs]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")

        for name, value in inputs.items():
            if name not in self.nodes:
                raise ValueError(f"Node {name} not found.")
            self.nodes[name].value = value

        if not self.sorted_nodes:
            self.topological_sort()
        name_list =[node.name for node in self.sorted_nodes]
        for node_name in name_list:
            if node_name in inputs:
                continue
            operation, *parents = adj.get(node_name, (None,))
            if operation is None:
                continue  # INPUT nodes are already handled
            parent_values = [self.nodes[p].value for p in parents]
            if operation == 'AND':
                self.nodes[node_name].value = int(all(parent_values))
            elif operation == 'OR':
                self.nodes[node_name].value = int(any(parent_values))
            elif operation == 'NOT':
                self.nodes[node_name].value = int(not parent_values[0])
            elif operation == 'OUTPUT':
                self.nodes[node_name].value = parent_values[0]
            elif operation == 'XOR':
                self.nodes[node_name].value = int(parent_values[0] ^ parent_values[1])
            else:
                raise ValueError(f"Unknown operation {operation} in {node_name}")

        outputs = [name for name, node in self.nodes.items()
                   if (outall and node.node_type != 'INPUT') or node.node_type == 'OUTPUT']
        return {name: self.nodes[name].value for name in outputs}

    def get_innode(self, node_name, adj=None):
        """
        Returns all input nodes that influence the given node.

        Args:
            node_name (str): The target node name.
            adj (dict, optional): The adjacency list representation of the DAG.

        Returns:
            list: A list of input node names.
        """
        if adj is None:
            adj = self.get_adj()
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist.")
        inputs = set()
        queue = deque([node_name])
        while queue:
            node = queue.popleft()
            parents = adj.get(node)
            if not parents or parents[0]=='INPUT':
                inputs.add(node)
            else:
                queue.extend(parents[1:])
        return list(inputs)

    def evaluate_timing(
            self,
            input_sequence: Dict[int, Dict[str, int]],
            time_steps: int = None
    ) -> Dict[str, List[int]]:
        """
        时序评估方法（优化版）

        改进点：
        1. 输入信号自动保持功能
        2. 增加时间步边界检查
        3. 优化父节点值获取逻辑
        """
        # 确定时间窗口并初始化
        time_steps = time_steps or (self.critical_path_delay() + 1)
        self._init_node_values(time_steps)

        # 处理输入信号（带保持逻辑）
        self._process_inputs(input_sequence, time_steps)

        # 按拓扑顺序计算各节点
        for node in self.sorted_nodes:
            if node.node_type == "INPUT":
                continue

            for t in range(time_steps):
                if t<node.delay:
                    continue
                # 计算有效输入时间
                effective_t = max(t - node.delay, 0)

                # 获取父节点值（优化版本）
                parent_values = self._get_parent_values(node.name, effective_t)

                # 执行逻辑运算
                node.values[t] = self._compute_gate_output(node.node_type, parent_values)

        return {name: node.values for name, node in self.nodes.items()}

    def _init_node_values(self, time_steps: int):
        """初始化所有节点时间序列值"""
        for node in self.nodes.values():
            node.values = [0] * time_steps

    def _process_inputs(self, input_sequence: Dict[int, Dict[str, int]], time_steps: int):
        """处理输入信号（带信号保持功能）"""
        # 第一阶段：应用显式输入值
        for t, inputs in input_sequence.items():
            if t >= time_steps:
                raise ValueError(f"Input time step {t} exceeds max {time_steps - 1}")
            for name, value in inputs.items():
                if name not in self.nodes or self.nodes[name].node_type != "INPUT":
                    raise ValueError(f"Invalid input node: {name}")
                self.nodes[name].values[t] = value

        # 第二阶段：填充未设置的时间步（信号保持）
        for node in self.nodes.values():
            if node.node_type == "INPUT":
                last_val = 0
                for t in range(time_steps):
                    if node.values[t] not in (0, 1):  # 处理未显式设置的情况
                        node.values[t] = last_val
                    else:
                        last_val = node.values[t]

    def _get_parent_values(self, node_name: str, effective_t: int) -> List[int]:
        """安全获取父节点值"""
        parent_values = []
        for parent_name in self.get_node_parents(node_name):
            parent_node = self.nodes[parent_name]

            # 边界检查
            if effective_t >= len(parent_node.values):
                raise IndexError(
                    f"Effective time {effective_t} out of range for {parent_name} "
                    f"(max: {len(parent_node.values) - 1})"
                )

            parent_values.append(parent_node.values[effective_t])
        return parent_values

    def _compute_gate_output(self, gate_type: str, inputs: List[int]) -> int:
        """带类型检查的逻辑门计算"""
        if not all(v in (0, 1) for v in inputs):
            raise ValueError(f"Invalid input values {inputs} for {gate_type} gate")

        if gate_type == "AND":
            return 1 if all(inputs) else 0
        elif gate_type == "OR":
            return 1 if any(inputs) else 0
        elif gate_type == "NOT":
            return 0 if inputs[0] else 1
        elif gate_type == "XOR":
            return inputs[0] ^ inputs[1]
        elif gate_type == "NAND":
            return 0 if all(inputs) else 1
        elif gate_type == "NOR":
            return 0 if any(inputs) else 1
        elif gate_type =='OUTPUT':
            return inputs[0]
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")


    def get_outnode(self):
        """Returns all output nodes in the DAG."""
        return [node for node in self.nodes.values() if node.node_type == 'OUTPUT']

    def get_all_inputs(self):
        """Returns all input nodes in the DAG."""
        return [node for node in self.nodes.values() if node.node_type == 'INPUT']
