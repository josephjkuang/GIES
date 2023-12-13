import xml.etree.ElementTree as ET

def create_annotation_xml(folder, filename, width, height, depth, object_data):
    annotation = ET.Element("annotation")

    folder_elem = ET.SubElement(annotation, "folder")
    folder_elem.text = folder

    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = filename

    size_elem = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size_elem, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size_elem, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size_elem, "depth")
    depth_elem.text = str(depth)

    segmented_elem = ET.SubElement(annotation, "segmented")
    segmented_elem.text = "0"

    object_elem = ET.SubElement(annotation, "object")

    name_elem = ET.SubElement(object_elem, "name")
    name_elem.text = object_data["name"]

    pose_elem = ET.SubElement(object_elem, "pose")
    pose_elem.text = "Unspecified"

    truncated_elem = ET.SubElement(object_elem, "truncated")
    truncated_elem.text = str(object_data["truncated"])

    occluded_elem = ET.SubElement(object_elem, "occluded")
    occluded_elem.text = str(object_data["occluded"])

    difficult_elem = ET.SubElement(object_elem, "difficult")
    difficult_elem.text = str(object_data["difficult"])

    bndbox_elem = ET.SubElement(object_elem, "bndbox")
    xmin_elem = ET.SubElement(bndbox_elem, "xmin")
    xmin_elem.text = str(object_data["bndbox"]["xmin"])
    ymin_elem = ET.SubElement(bndbox_elem, "ymin")
    ymin_elem.text = str(object_data["bndbox"]["ymin"])
    xmax_elem = ET.SubElement(bndbox_elem, "xmax")
    xmax_elem.text = str(object_data["bndbox"]["xmax"])
    ymax_elem = ET.SubElement(bndbox_elem, "ymax")
    ymax_elem.text = str(object_data["bndbox"]["ymax"])

    return annotation

def save_xml_to_file(xml_element, file_path):
    tree = ET.ElementTree(xml_element)
    tree.write(file_path)