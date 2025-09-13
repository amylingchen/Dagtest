from sympy import symbols
from z3 import *

from dag import DAG


def generate_minimal_e_outputs(adj,max_delay=0):
    num_steps = max_delay + 1  # 最长传播时间
    variables = {}
    for key, value in adj.items():
        variables[key] = [Bool(f'{key}_{t}') for t in range(num_steps)]

    # variables = {f"{node}_{t}": Bool(f"{node}_{t}") for node in adj for t in range(num_steps)}

    expression = generate_expression_time('H', time=max_delay, circuit=adj)
    solver = Solver()
    solver.add(expression == BoolVal(True))

    key_vars = [variables['A'][4], variables['B'][0], variables['B'][2], variables['C'][0], variables['C'][2]]

    # 生成所有可能的E输出组合
    solutions = []
    while solver.check() == sat:
        model = solver.model()
        model_set = {str(v): is_true(model[v]) for v in model.decls()}

        # 只保留关心的变量
        # filtered_solution = {var: model_set.get(var, False) for var in var_order}
        solutions.append(model_set)

        # 构造高效阻塞子句
        block = [var != model.evaluate(var) for var in key_vars]
        solver.add(Or(block))

    # 输出最终最小集合
    return solutions


def generate_expression_time(node, time, circuit, memo=None):

    if memo is None:
        memo = {}
    key = (node, time)
    if key in memo:
        return memo[key]

    gate_type, inputs, delay = circuit[node]

    # 处理输入节点
    if gate_type == "INPUT":
        # 显式定义输入变量的符号
        expr = Bool(f"{node}_{time}")

    # 处理输出节点
    elif gate_type == "OUTPUT":
        expr = generate_expression_time(inputs[0], time, circuit, memo)

    elif gate_type == "Delay":
        input_time = time - delay
        if input_time < 0:
            input_exprs = [BoolVal(False) for _ in inputs]
        expr = generate_expression_time(inputs[0], input_time, circuit, memo)
    # 处理逻辑门
    else:
        # 计算输入的有效时间步（处理负时间为False）
        input_time = time - delay
        if input_time < 0:
            input_exprs = [BoolVal(False) for _ in inputs]
        else:
            input_exprs = [
                generate_expression_time(inp, input_time, circuit, memo)
                for inp in inputs
            ]

        # 生成逻辑表达式
        if gate_type == "AND":
            expr = And(*input_exprs) if input_exprs else BoolVal(True)
        elif gate_type == "OR":
            expr = Or(*input_exprs) if input_exprs else BoolVal(False)
        elif gate_type == "NOT":
            expr = Not(input_exprs[0])
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    memo[key] = expr
    return expr


def minimal_circuit_inputs(adj,max_delay=6):
    num_steps = max_delay+1  # 最长传播时间
    variables={}
    for key,value in adj.items():
        variables[key] = [Bool(f'{key}_{t}') for t in range(num_steps)]

    variables = {f"{node}_{t}": Bool(f"{node}_{t}") for node in adj for t in range(num_steps)}

    expression =generate_expression_time('H',time=max_delay,circuit=adj)
    solver = Solver()
    solver.add(expression == BoolVal(True))
    # 目标：最小化 A, B, C 变量的数量
    key_vars = [variables['A_4'], variables['B_0'], variables['B_2'], variables['C_0'], variables['C_2']]
    opt = Optimize()
    opt.add(solver.assertions())  # 添加所有逻辑约束
    opt.minimize(Sum([If(var, 1, 0) for var in key_vars]))  # 目标函数：最小化输入变量

    solutions = []
    while opt.check() == sat:
        model = opt.model()
        sol = {str(v): is_true(model[v]) for v in model.decls()}
        solutions.append(sol)

        # 让 Z3 生成不同解
        opt.add(Or([var != model.evaluate(var) for var in key_vars]))

    return solutions

if __name__ == '__main__':

    adj1 = {
        'A': ('INPUT',[],0),
        'B': ('INPUT',[],0),
        'C': ('INPUT',[],0),
        'D': ('OR', ['B', 'C'], 2),
        'E': ('AND', ['A', 'D'], 1),
        'F': ('NOT', ['D'], 3),
        'G': ('OR', ['F', 'E'], 1),
        'H': ('OUTPUT', ['G'],0),
    }
    dag1 = DAG()
    dag1.create_dag(adj1)

    input_nodes = [node.name for node in dag1.nodes.values() if node.node_type=='INPUT']
    output_node = [node.name for node in dag1.nodes.values()  if node.node_type=='OUTPUT']

    out_exp=generate_expression_time(node='H',time=6,circuit=adj1)
    print(out_exp)
    solution = generate_minimal_e_outputs(adj1,max_delay=6)

    print("Minimal Solutions for E_6 = True:")
    for res in solution:
        print(res)

