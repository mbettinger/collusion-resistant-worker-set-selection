import graph_tool.all as gt

def emphasizeWorkers(F,workerIds):
    for v in F.vs:
        if v["name"] in workerIds:
            v["size"]=25
            v["shape"]="triangle"
        else:
            v["size"]=1
            v["shape"]="circle"
    return F

def drawGraph(graph,imgPath):
    G=graph.to_graph_tool(vertex_attributes={"color":"vector<float>","size":"int","shape":"string"},edge_attributes={"color":"vector<float>"})
    gt.graph_draw(G, output=imgPath,
                  vertex_fill_color=G.vertex_properties["color"],
                  vertex_shape=G.vertex_properties["shape"],
                  vertex_size=G.vertex_properties["size"],
                  edge_color=G.edge_properties["color"])