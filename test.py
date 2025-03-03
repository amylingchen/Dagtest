import pandas as pd

from dag import DAG

from utils import *

if __name__ == "__main__":
    #  创建DAG并添加节点（逻辑门）
    dag = DAG()
    # adj = {'MS1': None, 'MS2': None, 'MS3': None, 'MS4': None, 'MS5': None, 'MS6': None, 'MS7': None, 'MS8': None, 'MS9': None, 'MS10': None, 'MS11': None, 'MS12': None, 'MS13': None, 'MS14': None, 'MS15': None, 'MS16': ('OR', 'MS6', 'MS1', 'MS2', 'MS3'), 'MS17': ('NOT', 'MS1'), 'MS18': ('NOT', 'MS7'), 'MS19': ('OR', 'MS11', 'MS8', 'MS17'), 'MS20': ('NOT', 'MS9'), 'MS21': ('OR', 'MS17', 'MS3'), 'MS22': ('NOT', 'MS16'), 'MS23': ('OR', 'MS18', 'MS10'), 'MS24': ('NOT', 'MS17'), 'MS25': ('NOT', 'MS11'), 'MS26': ('AND', 'MS23', 'MS18'), 'MS27': ('AND', 'MS23', 'MS7', 'MS16'), 'MS28': ('AND', 'MS18', 'MS12', 'MS22', 'MS27'), 'MS29': ('NOT', 'MS3'), 'MS30': ('AND', 'MS28', 'MS18'), 'MS31': ('AND', 'MS13', 'MS30'), 'MS32': ('OR', 'MS20', 'MS31', 'MS8', 'MS30'), 'MS33': ('NOT', 'MS4'), 'MS34': ('NOT', 'MS29'), 'MS35': ('AND', 'MS5', 'MS4', 'MS6', 'MS13'), 'MS36': ('NOT', 'MS13'), 'MS37': ('AND', 'MS35', 'MS7', 'MS27', 'MS14'), 'MS38': ('AND', 'MS18', 'MS13', 'MS34'), 'MS39': ('AND', 'MS2', 'MS36', 'MS3'), 'MS40': ('OR', 'MS37', 'MS9', 'MS4', 'MS6'), 'MS41': ('NOT', 'MS15'), 'MS42': ('OR', 'MS39', 'MS40'), 'MS43': ('NOT', 'MS32'), 'MS44': ('NOT', 'MS15'), 'MS45': ('AND', 'MS22', 'MS15', 'MS5'), 'MS46': ('NOT', 'MS28'), 'MS47': ('NOT', 'MS24'), 'MS48': ('NOT', 'MS23'), 'MS49': ('NOT', 'MS5'), 'MS50': ('AND', 'MS34', 'MS30', 'MS19'), 'MS51': ('OR', 'MS34', 'MS37', 'MS31', 'MS25'), 'MS52': ('OR', 'MS45', 'MS13', 'MS44'), 'MS53': ('OUTPUT', 'MS46'), 'MS54': ('OUTPUT', 'MS47'), 'MS55': ('OUTPUT', 'MS48'), 'MS56': ('OUTPUT', 'MS49'), 'MS57': ('OUTPUT', 'MS50'), 'MS58': ('OUTPUT', 'MS51'), 'MS59': ('OUTPUT', 'MS52')}

    # adj ={
    #       'MS2': None,
    #       'MS3': None,
    #       'MS4': None,
    #       'MS7': ('NOT', ['MS3']),
    #       'MS8': ('OR', ['MS2', 'MS7']),
    #       # 'MS9': ('NOT', 'MS4'),
    #       'MS10': ('NOT', ['MS4']),
    #       'MS11': ('AND', ['MS8', 'MS10']),
    #       'MS14': ('OUTPUT', ['MS11']),
    #
    # }

    adj ={
          'MS2': None,
          'MS3': None,
          'MS4': None,
          'MS7': ('NOT', ['MS3']),
          'MS8': ('OR', ['MS2', 'MS3']),
          # 'MS9': ('NOT', 'MS4'),
          'MS10': ('NOT', ['MS4']),
          'MS11': ('OR', ['MS8', 'MS10']),
          'MS14': ('OUTPUT', ['MS11']),

    }


    dag.create_dag(adj)
    #拓扑排序（验证依赖顺序）
    # print("sorted order:", dag.topological_sort())

    # plot_dag_with_networkx(dag)

    # input= {'MS1': 1,'MS2': 1,'MS3':1,'MS4': 0,'MS5': 0,'MS6': 0,'MS7': 1,'MS8': 1, 'MS9': 0, 'MS10':0, 'MS11': 1, 'MS12': 0, 'MS13': 0, 'MS14': 1, 'MS15':0,'MS16': 0, 'MS17': 0, 'MS18': 0, 'MS19': 0, 'MS20': 0}
    # input = {'MS1': 1, 'MS2': 1, 'MS3': 1, 'MS4': 0, 'MS5': 0, 'MS6': 0, 'MS7': 1}
    # out =dag.evaluate( input)
    # print(out)

    outs= [node.name for node in dag.get_outnode()]
    out_node=outs[0]
    in_nodes=dag.get_innode(outs[0])
    cases=get_case_in_multiple(dag)
    outcases=create_out(cases, dag)
    save_cases(outcases,'dag')

    # save_dag_to_csv(dag)
    save_dag_to_csv(dag,'dag')




    # adj1 = {
    #     'S0':None,
    #     'S1':None,
    #     'T': None,
    #     'S3':('NOT','S0'),
    #     'S4':('NOT','S1'),
    #     'S5':('NOT','T'),
    #     # 'S6':('AND','S3','S4','S5'),
    #     # 'S7':('AND','S4','S0'),
    #     # 'S8':('AND','S1','S3'),
    #     # 'S9':('OR','S6','S7','S8'),
    #     # 'S10':('OUTPUT','S9'),
    #     'S11':('AND','S4','S5'),
    #     'S12':('XOR','S1','S0'),
    #     'S13':('OR','S11','S12'),
    #     'S14':('OUTPUT','S13'),
    #     # 'S15':('AND','S3','T'),
    #     # 'S16':('AND','S1','S3'),
    #     # 'S17':('OR','S15','S16'),
    #     # 'S18':('OUTPUT','S17'),
    #
    # }

    # adj1 = {
    #     'S0':None,
    #     'S1':None,
    #     'S2': None,
    #     'S3': ('OR', ['S1', 'S2'],2),
    #     'S4': ('AND', ['S0', 'S3'],1),
    #     'S5': ('NOT', ['S3'],3),
    #     'S6': ('OR', ['S5', 'S4'],1),
    #     'S8':('OUTPUT',['S6']),
    # }
    # dag1 = DAG()
    # dag1.create_dag(adj1)
    # #拓扑排序（验证依赖顺序）
    # input_seq = {
    #     0: {"S0": 0, "S1": 0, "S2": 1},
    #     1: {"S0": 1, "S1": 0, "S2": 0},
    #     2: {"S0": 1, "S1": 1, "S2": 1}
    # }
    # results = dag1.evaluate_timing(input_seq)
    #
    # # 打印输出结果
    # print("Time Step\t", "\t".join(sorted(results.keys())))
    # for t in range(len(results["S8"])):
    #     print(f"t={t}\t\t", end="")
    #     for node in sorted(results.keys()):
    #         print(f"{results[node][t]}", end="\t\t")
    #     print()
    # save_dag_to_csv(dag1,'dag1')






