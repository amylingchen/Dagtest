import random

n=15
# 生成节点列表
nodes = [f'MS{i}' for i in range(1,n)]
adjacency = {}
weights = [0.4, 0.4, 0.2]
for idx in range(len(nodes)):
    node = nodes[idx]

    if idx < 4:  # 输入节点
        adjacency[node] = None
    elif idx>=n-4:
        adjacency[node] = ('OUTPUT',nodes[idx-3])
    else:  # 逻辑门节点
        # 随机选择逻辑类型
        gate_type = random.choices(['AND', 'OR', 'NOT'],weights=weights, k=1)[0]
        num_inputs = 2 if gate_type in ['AND', 'OR'] else 1

        # 从编号更小的节点中选择输入
        available_inputs = nodes[:idx]
        inputs = random.sample(available_inputs, num_inputs)

        adjacency[node] = (gate_type,*inputs)

# 输出格式示例（前15个节点+最后5个）
print("邻接表示例：")
# for node in list(adjacency.keys())[:15] + nodes[-5:]:
#     inputs = adjacency[node]
#     print(f"{node}: {inputs}")
print(adjacency)