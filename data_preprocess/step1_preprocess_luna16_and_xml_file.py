import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob


random.seed(1321)
numpy.random.seed(1321)


def find_mhd_file(patient_id):
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "\\"
        src_paths = glob.glob(src_dir + "*.mhd")
        patient_id1=src_dir +patient_id+ ".mhd"
        if patient_id1 in src_paths:
                return patient_id1
    return None


def load_lidc_xml(xml_path, agreement_threshold=2, only_patient=None):
    pos_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    if only_patient is not None:
        if only_patient != patient_id:
            return None, None

    src_path = find_mhd_file(patient_id)
    if src_path is None:
        return None, None

    print(patient_id)
    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)

    #The following code obtains the center coordinates, diameter, and image features of the labeled nodule
    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin[2]
            z_center /= spacing[2]

            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)

    if agreement_threshold >= 0:
        filtered_lines = []
        for pos_line1 in extended_lines:
            id1 = pos_line1[1]
            x1 = pos_line1[2]
            y1 = pos_line1[3]
            z1 = pos_line1[4]
            d1 = pos_line1[5]
            overlaps = 0
            for pos_line2 in extended_lines:
                id2 = pos_line2[1]
                if id1 == id2:
                    continue
                x2 = pos_line2[2]
                y2 = pos_line2[3]
                z2 = pos_line2[4]
                d2 = pos_line2[5]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)

        pos_lines_extended = filtered_lines

    df_annos = pandas.DataFrame(pos_lines_extended, columns=["uid","anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore","sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
    df_annos.to_csv("data/posline9/" + patient_id+ "_annos_pos_lidc.csv", index=False)
    return pos_lines_extended,extended_lines

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def process_images(delete_existing=False, only_process_patient=None):
    if delete_existing and os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.LUNA16_EXTRACTED_IMAGE_DIR)

    if not os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.LUNA16_EXTRACTED_IMAGE_DIR)
        os.mkdir(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/")

    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        src_paths = glob.glob(src_dir + "*.mhd")

        if only_process_patient is None and True:
            pool = multiprocessing.Pool(settings.WORKER_POOL_SIZE)
            pool.map(process_image, src_paths)
        else:
            for src_path in src_paths:
                print(src_path)
                if only_process_patient is not None:
                    if only_process_patient not in src_paths:
                        continue
                process_image(src_path)


def process_lidc_annotations(only_patient=None, agreement_threshold=2):
    file_no = 0
    for anno_dir in [d for d in glob.glob("resources/luna16_annotations/*") if os.path.isdir(d)]:

        xml_paths = glob.glob(anno_dir + "/*.xml")

        for xml_path in xml_paths:
            print(file_no, ": ",  xml_path)
            load_lidc_xml(xml_path=xml_path, only_patient=only_patient, agreement_threshold=agreement_threshold)

if __name__ == "__main__":
     if True:
        process_images(delete_existing=False, only_process_patient=None)
        #Traverse all MHD files and convert MHD files into CT slices one by one

     if True:
         process_lidc_annotations(only_patient=None, agreement_threshold=2)
         #Get structured features  in xml file


