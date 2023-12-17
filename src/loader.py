import glfw
import struct
import math
import io
import pygltflib

import numpy as np

from PIL import Image
from pygltflib import GLTF2

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


# Window size
width, height = 512, 512


# Initialize GLFW for headless rendering
def initialize_headless_opengl():
    if not glfw.init():
        print("Failed to initialize GLFW")
        return None

    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(800, 600, "Hidden Window", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return None

    glfw.make_context_current(window)
    return window


def terminate_opengl(window):
    glfw.destroy_window(window)
    glfw.terminate()


def create_offscreen_buffer():
    # Set up FBO for off-screen rendering

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Create a texture to render to
    texture = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

    # Attach texture to FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

    # Check FBO status
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        print("Error setting up FBO")
        return None

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo, texture


def take_screenshot(fbo, filename):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = Image.frombytes("RGB", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image.save(filename)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


def calculate_bounding_box(vertices):
    min_x = min(vertices, key=lambda v: v[0])[0]
    max_x = max(vertices, key=lambda v: v[0])[0]
    min_y = min(vertices, key=lambda v: v[1])[1]
    max_y = max(vertices, key=lambda v: v[1])[1]
    min_z = min(vertices, key=lambda v: v[2])[2]
    max_z = max(vertices, key=lambda v: v[2])[2]

    return (min_x, max_x, min_y, max_y, min_z, max_z)


def calculate_object_center(bbox):
    centerX = (bbox[0] + bbox[1]) / 2.0
    centerY = (bbox[2] + bbox[3]) / 2.0
    centerZ = (bbox[4] + bbox[5]) / 2.0
    return (centerX, centerY, centerZ)


def setup_view(angle, bbox, fov=45.0):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, (width / height), 0.1, 100.0)

    object_width = bbox[1] - bbox[0]
    object_height = bbox[3] - bbox[2]
    object_depth = bbox[5] - bbox[4]

    # Calculate the distance
    distance = max(object_width, object_height, object_depth) / (2 * math.tan(math.radians(fov / 2)))

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    x, y = angle
    cameraX, cameraY, cameraZ = distance * math.sin(math.radians(x)), distance, distance * math.cos(math.radians(y))

    centerX, centerY, centerZ = calculate_object_center(bbox)

    gluLookAt(
        cameraX + centerX, cameraY + centerY, cameraZ + centerZ,  # Camera position
        centerX, centerY, centerZ,                                 # Look at the object's center
        0, 1, 0                                                    # Up vector
    )


def load_texture(filename):
    # Load a texture from a file
    image = Image.open(filename)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = np.array(list(image.getdata()), np.uint8)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    return texture


def load_mtl(filename):
    # Load material data from .mtl file
    materials = {}
    current_mtl = None

    if not os.path.exists(filename):
        return materials

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('newmtl'):
                _, name = line.split()
                current_mtl = name
                materials[current_mtl] = {'texture': None}
            elif line.startswith('map_Kd') and current_mtl:
                # Load texture map
                _, tex_file = line.split()
                texture = load_texture(tex_file)
                materials[current_mtl]['texture'] = texture

    return materials


def load_obj(filename):
    vertices = []
    texcoords = []
    normals = []
    faces = []
    materials = {}
    current_mtl = None

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()[:4]
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith('vt '):
                _, u, v = line.split()[:3]
                texcoords.append((float(u), float(v)))
            elif line.startswith('vn '):
                _, nx, ny, nz = line.split()[:4]
                normals.append((float(nx), float(ny), float(nz)))
            elif line.startswith('usemtl'):
                _, mtl_name = line.strip().split()
                current_mtl = mtl_name
            elif line.startswith('f '):
                face = line.split()[1:]
                face_vertices = []
                for vertex in face:
                    parts = vertex.split('/')
                    v = int(parts[0]) - 1
                    t = int(parts[1]) - 1 if len(parts) > 1 and parts[1] else -1
                    n = int(parts[2]) - 1 if len(parts) > 2 and parts[2] else -1
                    face_vertices.append((v, t, n, current_mtl))
                faces.append(face_vertices)
            elif line.startswith('mtllib'):
                mtl_file = os.path.join(os.path.dirname(filename), line.split()[-1].strip())
                materials.update(load_mtl(mtl_file))

    return vertices, texcoords, normals, faces, materials


def extract_glb_binary(file_path):
    # Extract the binary chunk from the GLB file
    # The actual implementation depends on the GLB file structure
    with open(file_path, 'rb') as f:
        # Read the header
        magic, version, length = struct.unpack('<4sII', f.read(12))
        if magic != b'glTF':
            raise ValueError("File is not a GLB format")

        # Read chunks
        while f.tell() < length:
            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
            if chunk_type == 0x4E4F534A:  # 'JSON'
                f.seek(chunk_length, os.SEEK_CUR)  # Skip JSON chunk
            elif chunk_type == 0x004E4942:  # 'BIN\0'
                return f.read(chunk_length)  # Read binary chunk

    raise ValueError("No binary chunk found in GLB file")


def load_glb(file_path):
    gltf = GLTF2().load(file_path)

    vertices = []
    texcoords = []
    normals = []
    faces = []
    materials = {}

    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            v_buffer = extract_data(gltf, primitive.attributes.POSITION)
            n_buffer = extract_data(gltf, primitive.attributes.NORMAL)
            t_buffer = extract_data(gltf, primitive.attributes.TEXCOORD_0)
            f_buffer = extract_data(gltf, primitive.indices)

            vertices.extend(v_buffer)
            normals.extend(n_buffer)
            texcoords.extend(t_buffer)
            faces.extend(f_buffer)

            if primitive.material is not None:
                material_data = process_material(gltf, primitive.material)
                materials[primitive.material] = material_data

    return vertices, texcoords, normals, faces, materials


def extract_data(gltf, accessor_index):
    accessor = gltf.accessors[accessor_index]
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]

    # Assuming the buffer data is embedded in the GLB file
    # This part needs modification if the buffer data is stored externally
    byte_offset = bufferView.byteOffset if bufferView.byteOffset else 0
    byte_length = bufferView.byteLength
    binary_data = buffer[byte_offset:byte_offset + byte_length]

    # Interpret the binary data based on the accessor component type
    if accessor.componentType == pygltflib.FLOAT:
        data_type = np.float32
    elif accessor.componentType == pygltflib.UNSIGNED_INT:
        data_type = np.uint32
    # Add handling for other component types as needed

    num_components = len(pygltflib.TYPEMAP[accessor.type])
    data = np.frombuffer(binary_data, dtype=data_type)
    data = data.reshape(-1, num_components)

    return data.tolist()


def process_material(gltf, material_index):
    material = gltf.materials[material_index]
    material_data = {}

    # Extract base color
    if material.pbrMetallicRoughness and material.pbrMetallicRoughness.baseColorFactor:
        material_data['baseColor'] = material.pbrMetallicRoughness.baseColorFactor

    # Here you can add more material properties extraction as needed

    return material_data


def load_texture_from_uri(uri):
    # Load the image using PIL
    image = Image.open(uri)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image vertically
    img_data = np.array(list(image.getdata()), np.uint8)

    # Generate an OpenGL texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture


def load_texture_from_buffer_view(gltf, buffer_view_index):
    bufferView = gltf.bufferViews[buffer_view_index]
    buffer = gltf.buffers[bufferView.buffer]

    # Access binary data from the buffer
    byte_offset = bufferView.byteOffset if bufferView.byteOffset else 0
    byte_length = bufferView.byteLength
    binary_data = buffer.data[byte_offset:byte_offset+byte_length]

    # Convert binary data to an image
    image = Image.open(io.BytesIO(binary_data))
    image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image vertically
    img_data = np.array(list(image.getdata()), np.uint8)

    # Generate an OpenGL texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture


def setup_lighting():
    light_position = [10.0, 10.0, 10.0, 1.0]  # Positional light
    light_diffuse = [1.0, 1.0, 1.0, 1.0]      # White diffuse light
    light_ambient = [0.1, 0.1, 0.1, 1.0]      # Low intensity ambient light

    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)


def set_default_material():
    material_diffuse = [0.8, 0.8, 0.8, 1.0]  # Grayish material
    glMaterialfv(GL_FRONT, GL_DIFFUSE, material_diffuse)


def draw_scene(vertices, texcoords, normals, faces, materials):
    glEnable(GL_TEXTURE_2D)
    for face in faces:
        # Set material here if available
        glBegin(GL_POLYGON)
        for vertex in face:
            if normals:
                glNormal3fv(normals[vertex[2]])  # Set normal vector
            if texcoords:
                glTexCoord2fv(texcoords[vertex[1]])
            glVertex3fv(vertices[vertex[0]])
        glEnd()
    glDisable(GL_TEXTURE_2D)


def create_view_perspective(file_path: str):
    # # Initialize GLFW for headless rendering
    window = initialize_headless_opengl()
    if not window:
        return

    # [Load model and set up OpenGL context, if necessary]
    fbo, renderedTexture = create_offscreen_buffer()
    if not fbo:
        return

    # Enable texture mapping and other necessary settings
    glEnable(GL_LIGHTING)  # Enable lighting
    glEnable(GL_LIGHT0)    # Enable light 0
    glEnable(GL_TEXTURE_2D)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)

    # Load 3D model with texture and material support
    global vertices, texcoords, normals, faces, materials
    # check extension of file
    if file_path.endswith('.obj'):
        vertices, texcoords, normals, faces, materials = load_obj(file_path)
    elif file_path.endswith('.glb'):
        print("WARN: Not working properly!")
        vertices, texcoords, normals, faces, materials = load_glb(file_path)
    else:
        raise ValueError('File extension not supported')
    bbox = calculate_bounding_box(vertices)

    # Rendering and capturing loop
    for i, angle in enumerate([(0, 3), (93, 0), (50, 110), (260, 0)]):
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, width, height)

        # Clear the color and depth buffers
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set the clear color (black in this case)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up the camera for the specified viewing angle
        setup_view(angle, bbox, fov=45.0)
        # Render the scene
        setup_lighting()
        set_default_material()
        draw_scene(vertices, texcoords, normals, faces, materials)
        take_screenshot(fbo, f"outputs/screenshot_{i}.png")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # Clean up resources
    glDeleteTextures(1, [renderedTexture])
    glDeleteFramebuffers(1, [fbo])
    terminate_opengl(window)


if __name__ == "__main__":
    create_view_perspective('/Users/xpitfire/Downloads/32-mercedes-benz-gls-580-2020/uploads_files_2787791_Mercedes+Benz+GLS+580.obj')
