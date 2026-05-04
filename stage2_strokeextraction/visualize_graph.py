import json
import matplotlib.pyplot as plt
import sys
import os

def visualize_graph(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Invert Y axis because image coordinates typically have Y going down
    ax.invert_yaxis()
    
    # Plot edges
    for edge in data.get('edges', []):
        pixels = edge.get('pixels', [])
        if pixels:
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            ax.plot(xs, ys, color='blue', alpha=0.6, linewidth=2)
    
    # Plot nodes
    for node in data.get('nodes', []):
        x = node['x']
        y = node['y']
        
        # Color based on type
        if node.get('type') == 'endpoint':
            color = 'red'
            marker = 'o'
        else:
            color = 'green'
            marker = 's'
            
        ax.scatter(x, y, color=color, marker=marker, s=50, zorder=5)
        ax.text(x+3, y+3, str(node['id']), fontsize=9, color='black', weight='bold')

    # Get image shape to set limits
    shape = data.get('image_shape', [256, 256])
    ax.set_xlim(0, shape[1])
    ax.set_ylim(shape[0], 0) # y-axis is inverted
    
    ax.set_title(f"Graph Visualization: {os.path.basename(json_path)}")
    ax.set_aspect('equal')
    
    plt.savefig(output_path)
    print(f"Graph saved to {output_path}")

if __name__ == '__main__':
    json_file = sys.argv[1]
    out_file = sys.argv[2]
    visualize_graph(json_file, out_file)
