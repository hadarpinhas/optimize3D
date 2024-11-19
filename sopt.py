import open3d as o3d
import numpy as np

def create_shape_from_text(text):
    """Create a 3D shape based on the input text description."""
    
    # Simple text parsing logic to extract shape type and dimensions
    text = text.lower()
    
    if "cube" in text:
        # Example: "Create a cube of size 1"
        size = float([word for word in text.split() if word.replace('.', '', 1).isdigit()][0])
        return create_cube(size)
    
    elif "sphere" in text:
        # Example: "Create a sphere with radius 2"
        radius = float([word for word in text.split() if word.replace('.', '', 1).isdigit()][0])
        return create_sphere(radius)
    
    elif "cylinder" in text:
        # Example: "Create a cylinder with radius 1 and height 2"
        dimensions = [float(word) for word in text.split() if word.replace('.', '', 1).isdigit()]
        radius, height = dimensions[0], dimensions[1]
        return create_cylinder(radius, height)
    
    else:
        raise ValueError("Unsupported shape or format in the text.")

def create_cube(size):
    """Generate a cube of given size."""
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    mesh.compute_vertex_normals()
    return mesh

def create_sphere(radius):
    """Generate a sphere with the given radius."""
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.compute_vertex_normals()
    return mesh

def create_cylinder(radius, height):
    """Generate a cylinder with given radius and height."""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh.compute_vertex_normals()
    return mesh

def visualize_3d_model(mesh):
    """Visualize the generated 3D model."""
    o3d.visualization.draw_geometries([mesh])

def main():
    # Sample input text to create a 3D model
    input_text = input("Enter a description of the 3D shape: ")
    
    try:
        # Create the 3D shape from the input text
        shape = create_shape_from_text(input_text)
        
        # Visualize the shape
        visualize_3d_model(shape)
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
