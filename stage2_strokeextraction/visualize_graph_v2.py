import json
import matplotlib.pyplot as plt
import sys
import os

def visualize_graph_v2(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Invert Y axis because image coordinates typically have Y going down
    ax.invert_yaxis()
    
    # Plot edges (strokes)
    for edge in data.get('edges', []):
        pixels = edge.get('pixels', [])
        if pixels:
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            ax.plot(xs, ys, color='blue', alpha=0.6, linewidth=2)
    

    # Get image shape to set limits
    shape = data.get('image_shape', [256, 256])
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0) # y-axis is inverted
    
    # Clean up the output by removing the axes grid and labels entirely 
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Save the output image with tighter bounding box
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Graph saved to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python visualize_graph_V2.py <input.json> <output.png>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    out_file = sys.argv[2]
    visualize_graph_v2(json_file, out_file)
