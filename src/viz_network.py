import dash
from dash import html, dcc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output
import pandas as pd



def process_node_scores(node_scores, TFlist):
    node_scores["isTFpositive"]=["None"]*len(node_scores)
    for index,row in node_scores.iterrows():
        if row['J-t2-(LMP to CLP)']<0:
            isPositive="neg"
        else:
            isPositive="pos"
        if row["Genes"] in TFlist:
            isTF="True"
        else:
            isTF="False"
            
        annot=isTF+"-"+isPositive
        node_scores.at[index,"isTFpositive"]=annot
    return node_scores

def process_edge_list(edge_list):
    edge_list["edgeType"]=["None"]*len(edge_list)
    for index,row in edge_list.iterrows():
        if row['t2-(LMP to CLP)']<0:
            edge_list.at[index, "edgeType"]="Negative"
        else:
            edge_list.at[index, "edgeType"]="Positive"
    return edge_list


def create_draggable_network_with_slider(edge_list, node_scores, node_names=None, layout_type="cose"):
    """
    Create a network visualization with a time slider to dynamically update edge strengths and node Jacobian values.
    """
    # Time points from edge_list and node_scores columns
    time_points = [col for col in edge_list.columns if col.startswith("t")]
    
    # Subset the edge list based on node_names
    if node_names:
        edge_list = edge_list[
            (edge_list["TF"].isin(node_names)) | (edge_list["Target"].isin(node_names))
        ]
    
    # Create Dash App
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        # Slider for time selection
        html.Div([
            html.Label(
                "Select Time Point", 
                style={"font-size": "24px", "font-weight": "bold", "margin-bottom": "10px"}  # Increase font size and bold
            ),
            dcc.Slider(
                id="time-slider",
                min=0,
                max=len(time_points) - 1,
                step=1,
                marks={
                    i: {"label": time, "style": {"font-size": "22px", "font-weight": "bold"}} 
                    for i, time in enumerate(time_points)
                },  # Styling for slider marks
                value=0  # Default to the first time point
            )
        ], style={"margin-bottom": "20px"}),
        
        # Cytoscape Network Visualization
        cyto.Cytoscape(
            id="cytoscape-network",
            layout={"name": layout_type},  # Use specified layout type
            style={"width": "100%", "height": "460px"},
            stylesheet=[]  # Stylesheet will be updated dynamically
        )
    ])
    
    @app.callback(
        [Output("cytoscape-network", "elements"),
         Output("cytoscape-network", "stylesheet")],
        [Input("time-slider", "value")]
    )
    def update_network(time_index):
        # Get the selected time point
        selected_time = time_points[time_index]
        jacobian_col = f"J-{selected_time}"
        
        # Create a dictionary for quick lookup of node scores and properties
        score_dict = node_scores.set_index("Genes").to_dict(orient="index")
        
        # Generate Cytoscape elements
        nodes = [
            {
                "data": {
                    "id": node,
                    "label": node,
                    "jacobian": score_dict.get(node, {}).get(jacobian_col, 0),
                    "abs_jacobian": abs(score_dict.get(node, {}).get(jacobian_col, 0)),
                    "isTFpositive": score_dict.get(node, {}).get("isTFpositive", "False-pos")
                }
            }
            for node in set(edge_list["TF"]).union(edge_list["Target"])
        ]
        edges = [
            {
                "data": {
                    "source": row["TF"], 
                    "target": row["Target"], 
                    "strength": row[selected_time],
                    "edgeType": row["edgeType"],
                    "arrow_shape": "tee" if row[selected_time] < 0 else "vee"
                }
            } 
            for _, row in edge_list.iterrows()
        ]
        
        elements = nodes + edges
        
        # Update stylesheet based on the selected time point
        stylesheet = [
            # Style for TF nodes (rectangles) with positive Jacobian
            {
                "selector": '[isTFpositive = "True-pos"]', 
                "style": {
                    "shape": "round-rectangle",
                    "label": "data(label)",
                    "width": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "height": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "background-color": "mapData(jacobian, 0, 1, #cee9f0, #003878)"
                }
            },
            # Style for TF nodes (rectangles) with negative Jacobian
            {
                "selector": '[isTFpositive = "True-neg"]', 
                "style": {
                    "shape": "round-rectangle",
                    "label": "data(label)",
                    "width": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "height": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "background-color": "mapData(jacobian, -1, 0, #912020, #e3caca)"
                }
            },
            # Style for non-TF nodes (circles) with positive Jacobian
            {
                "selector": '[isTFpositive = "False-pos"]', 
                "style": {
                    "shape": "ellipse",
                    "label": "data(label)",
                    "width": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "height": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "background-color": "mapData(jacobian, 0, 1, #c5dce3, #003878)"
                }
            },
            # Style for non-TF nodes (circles) with negative Jacobian
            {
                "selector": '[isTFpositive = "False-neg"]', 
                "style": {
                    "shape": "ellipse",
                    "label": "data(label)",
                    "width": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "height": "mapData(abs_jacobian, 0, 1, 20, 50)",
                    "background-color": "mapData(jacobian, -1, 0, #912020, #e3caca)"
                }
            },
            # Style for edges
            {
                "selector": '[edgeType = "Negative"]', 
                "style": {
                    "width": "mapData(strength, -1, 0, 10, 1.5)",
                    "line-color": "mapData(strength, -1, 0, #912020, #e3caca)",
                    "curve-style": "bezier",
                    "target-arrow-shape": "data(arrow_shape)",
                    "target-arrow-color": "mapData(strength, -1, 0, #912020, #e3caca)"
                }
            },
            {
                "selector": '[edgeType = "Positive"]', 
                "style": {
                    "width": "mapData(strength, 0, 1, 1.5, 10)",
                    "line-color": "mapData(strength, 0, 1, #ebf6fa, #003878)",
                    "curve-style": "bezier",
                    "target-arrow-shape": "data(arrow_shape)",
                    "target-arrow-color": "mapData(strength, 0, 1, #ebf6fa, #003878)"
                }
            }
        ]
        
        return elements, stylesheet
    
    return app

app = create_draggable_network_with_slider(edge_list, node_scores_processed, node_names=all_nodes, layout_type="circle")
app.run_server(debug=True, use_reloader=False)


