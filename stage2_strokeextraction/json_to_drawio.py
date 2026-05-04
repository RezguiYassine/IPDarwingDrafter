import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import sys
import os

def json_to_drawio(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    mxfile = ET.Element('mxfile')
    diagram = ET.SubElement(mxfile, 'diagram', id='diagram_1', name='Graph')
    
    mxGraphModel = ET.SubElement(diagram, 'mxGraphModel', dx='1000', dy='1000', grid='1', gridSize='10', 
                                 guides='1', tooltips='1', connect='1', arrows='1', fold='1', 
                                 page='1', pageScale='1', pageWidth='827', pageHeight='1169', math='0', shadow='0')
    root = ET.SubElement(mxGraphModel, 'root')
    
    # Required base cells
    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')
    
    # Add nodes
    # For a node, we make a small square/circle
    for node in data.get('nodes', []):
        node_id = f"node_{node['id']}"
        node_type = node.get('type', '')
        
        # Style depending on type
        if node_type == 'endpoint':
            # Circle format (red)
            style = "ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor=#f8cecc;strokeColor=#b85450;fontColor=#000000;"
            size = 16
        else:
            # Junction format (square/green)
            style = "whiteSpace=wrap;html=1;aspect=fixed;fillColor=#d5e8d4;strokeColor=#82b366;fontColor=#000000;"
            size = 14
            
        # Center the shape over the coordinate
        x = node['x'] - size / 2
        y = node['y'] - size / 2
        
        mx_cell = ET.SubElement(root, 'mxCell', id=node_id, value=str(node['id']), style=style, vertex='1', parent='1')
        ET.SubElement(mx_cell, 'mxGeometry', x=str(x), y=str(y), width=str(size), height=str(size), **{'as': 'geometry'})

    # Add edges
    for edge in data.get('edges', []):
        edge_id = f"edge_{edge['id']}"
        source_id = f"node_{edge['source']}"
        target_id = f"node_{edge['target']}"
        
        # We use curved=1 to smoothly interpolate between the path's waypoints!
        style = "endArrow=none;html=1;rounded=0;curved=1;strokeColor=#6c8ebf;strokeWidth=2;"
        
        mx_cell = ET.SubElement(root, 'mxCell', id=edge_id, value='', style=style, edge='1', parent='1', 
                                source=source_id, target=target_id)
        
        geometry = ET.SubElement(mx_cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
        
        # Draw.io routing waypoints
        # We use the 'smooth_pts' because 'pixels' has way too many points for efficient vector rendering.
        # We also skip the first and last point because draw.io anchors the ends to the nodes directly.
        smooth_pts = edge.get('smooth_pts', [])
        if len(smooth_pts) > 2:
            array = ET.SubElement(geometry, 'Array', **{'as': 'points'})
            for pt in smooth_pts[1:-1]:
                ET.SubElement(array, 'mxPoint', x=str(pt[0]), y=str(pt[1]))

    # Pretty print XML
    xmlstr = minidom.parseString(ET.tostring(mxfile, encoding='utf-8')).toprettyxml(indent="  ")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xmlstr)
        
    print(f"Successfully generated Draw.io file: {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python json_to_drawio.py <input.json> <output.drawio>")
        sys.exit(1)
        
    json_to_drawio(sys.argv[1], sys.argv[2])
