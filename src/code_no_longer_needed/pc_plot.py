import pandas as pd
from pyvis.network import Network
import os

def generate_final_focus_network():
    # 1. 路径设置 (确保指向你的 TCGA 边列表文件)
    edge_file = "/data/zliu/Path_MoE/data/causal_analysis/TCGA_BRCA/TCGA_Cholesterol_Causal_Edges.csv"
    output_html = "/data/zliu/Path_MoE/data/causal_analysis/TCGA_BRCA/Minimalist_Focus_Network_Final.html"
    
    if not os.path.exists(edge_file):
        print(f"错误: 找不到边列表文件 {edge_file}")
        return
        
    df_edges = pd.read_csv(edge_file)
    
    # 2. 初始化网络：有向图，关闭所有默认花哨颜色
    # directed=True 确保箭头存在
    net = Network(height="950px", width="100%", bgcolor="#ffffff", font_color="#333333", directed=True)
    
    # 3. 核心视觉与交互配置 (Fix 1 & Fix 2)
    # shape: "circle" 让文字强行进入圆圈内部
    # selectConnectedEdges + highlight 组合实现“选中高亮，其余变灰”
    net.set_options("""
    {
      "interaction": {
        "hover": true,
        "hoverConnectedEdges": true,
        "selectConnectedEdges": true,
        "navigationButtons": true,
        "multiselect": true,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
      },
      "nodes": {
        "shape": "circle",
        "borderWidth": 1.5,
        "size": 25,
        "font": {
          "size": 14,
          "face": "arial",
          "align": "center"
        },
        "color": {
          "background": "#d5dbdb",
          "border": "#7f8c8d",
          "highlight": {"background": "#3498db", "border": "#2980b9"},
          "hover": {"background": "#d5dbdb", "border": "#2980b9"}
        }
      },
      "edges": {
        "color": {
          "color": "#bdc3c7",
          "highlight": "#3498db", 
          "hover": "#7f8c8d",
          "inherit": false
        },
        "width": 1,
        "smooth": {"type": "continuous"},
        "arrows": {"to": {"enabled": true, "scaleFactor": 1.2}}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -150,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 200}
      }
    }
    """)

    # 4. 提取节点并添加
    nodes = list(set(df_edges['Source'].tolist() + df_edges['Target'].tolist()))
    print(f"正在构建包含 {len(nodes)} 个基因的交互网络...")
    
    for node in nodes:
        net.add_node(node, label=node, title=f"基因: {node}")

    # 5. 添加连线
    for _, row in df_edges.iterrows():
        u, v, t = row['Source'], row['Target'], row['Type']
        
        if t == 'Directed':
            # 有向边：实线
            net.add_edge(u, v, width=1.5)
        else:
            # 无向边：虚线，不带箭头
            net.add_edge(u, v, width=1, dashes=True, arrows={'to': {'enabled': False}})

    # 6. 保存网页
    net.save_graph(output_html)
    print(f"\n[成功] 最终版交互网页已生成: {output_html}")
    print("视觉特性：")
    print("1. 基因名已锁定在圆圈内部 (Shape: Circle)")
    print("2. 点击任意基因，不相关的基因和边都会整体淡化变灰。")

# --- 关键修正：确保这里的函数名和上面定义的一致 ---
if __name__ == "__main__":
    generate_final_focus_network()