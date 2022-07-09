import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import pandas as pd
import ntpath
import math
import glob


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


def process_pos_annotations_patient(src_path, patient_id):
    global cubes_1vs45_pos,cubes_1vs3vs5_pos ,cubes_1vs5_pos,cubes_12vs45_pos,cubes_1vs3vs5_neg,cubes_1vs5_neg,cubes_1vs45_neg,cubes_12vs45_neg,y_1vs45_pos,y_1vs3vs5_pos ,y_1vs5_pos,y_12vs45_pos,y_1vs3vs5_neg,y_1vs5_neg,y_1vs45_neg,y_12vs45_neg,cubes_1vs3vs5_3,y_1vs3vs5_3
    df_node = pandas.read_csv("resources/luna16_annotations/annotations.csv")

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    img_array[img_array==-2048]=-1024
    print("Img array: ", img_array.shape)


    df_patient = df_node[df_node["seriesuid"] == patient_id]
    print("Annos in luna16 : ", len(df_patient))
    if not len(df_patient)==0 :

        origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        flip_direction_x = False
        flip_direction_y = False
        if round(direction[0]) == -1:
            origin[0] *= -1
            direction[0] = 1
            flip_direction_x = True
            print("Swappint x origin")
        if round(direction[4]) == -1:
            origin[1] *= -1
            direction[4] = 1
            flip_direction_y = True
            print("Swappint y origin")
        assert abs(sum(direction) - 3) < 0.01

        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

        df_patient = df_node[df_node["seriesuid"] == patient_id]
        nodule_pos=0
        nodule_neg=0
        nodule_3=0
        nodule_no_fusions=[]

        for index, annotation in df_patient.iterrows():
            node_x = annotation["coordX"]
            if flip_direction_x:
                node_x *= -1
            node_y = annotation["coordY"]
            if flip_direction_y:
                node_y *= -1
            node_z = annotation["coordZ"]
            diam_mm = annotation["diameter_mm"]
            center_float = numpy.array([node_x, node_y, node_z])
            center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
            center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
            diameter_pixels = diam_mm / settings.TARGET_VOXEL_MM
            diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

            x1=center_float_percent[0]
            y1=center_float_percent[1]
            z1=center_float_percent[2]
            d1=diameter_percent
            character=[]

            cube=get_cube_from_img( patient_imgs,center_float_rescaled[0],center_float_rescaled[1],center_float_rescaled[2],32)
            nodule_extended=pandas.read_csv("data/posline9/"+patient_id+"_annos_pos_lidc.csv")

            for index ,row in nodule_extended.iterrows():
                x2=row["coord_x"]
                y2=row["coord_y"]
                z2=row["coord_z"]
                d2=row["diameter"]

                dist=math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < (d1+d2)/2:
                    new_row=row[6:15].tolist()
                    new_row.append(round(annotation["diameter_mm"],2))
                    character.append(new_row)
            character=numpy.array(character)
            if not len(character)==0:
                character=numpy.round(character.mean(axis=0))
                print("nudule malignacne",character[0])

                if character[0]==5:
                    nodule_pos +=1
                    cubes_1vs45_pos +=1
                    cubes_1vs3vs5_pos +=1
                    cubes_1vs5_pos += 1
                    cubes_12vs45_pos += 1
                    y_1vs45_pos += 1
                    y_1vs5_pos += 1
                    y_12vs45_pos += 1
                    y_1vs3vs5_pos += 1

                    cubes_1vs45.append(cube)
                    cubes_1vs5.append(cube)
                    cubes_12vs45.append(cube)
                    cubes_1vs3vs5.append(cube)
                    y_1vs45.append(1)
                    y_1vs5.append(1)
                    y_12vs45.append(1)
                    y_1vs3vs5.append(1)
                    chara_1vs45.append(character)
                    chara_1vs5.append(character)
                    chara_12vs45.append(character)
                    chara_1vs3vs5.append(character)
                elif character[0]==4:
                    nodule_pos +=1
                    cubes_1vs45_pos += 1
                    cubes_12vs45_pos += 1
                    y_1vs45_pos += 1
                    y_12vs45_pos += 1

                    cubes_1vs45.append(cube)
                    cubes_12vs45.append(cube)
                    y_1vs45.append(1)
                    y_12vs45.append(1)
                    chara_1vs45.append(character)
                    chara_12vs45.append(character)
                elif character[0]==3:
                    nodule_3+=1
                    cubes_1vs3vs5_3+=1
                    y_1vs3vs5_3 += 1

                    cubes_1vs3vs5.append(cube)
                    y_1vs3vs5.append(3)
                    chara_1vs3vs5.append(character)
                elif character[0]==2:
                    nodule_neg+=1
                    cubes_12vs45_neg+=1
                    y_12vs45_neg += 1

                    cubes_12vs45.append(cube)
                    y_12vs45.append(0)
                    chara_12vs45.append(character)
                elif character[0]==1:
                    nodule_neg+=1
                    cubes_1vs45_neg += 1
                    cubes_1vs3vs5_neg += 1
                    cubes_1vs5_neg += 1
                    cubes_12vs45_neg += 1
                    y_1vs45_neg += 1
                    y_1vs5_neg += 1
                    y_12vs45_neg += 1
                    y_1vs3vs5_neg += 1

                    cubes_1vs45.append(cube)
                    cubes_1vs5.append(cube)
                    cubes_12vs45.append(cube)
                    cubes_1vs3vs5.append(cube)
                    y_1vs45.append(0)
                    y_1vs5.append(0)
                    y_12vs45.append(0)
                    y_1vs3vs5.append(0)
                    chara_1vs45.append(character)
                    chara_1vs5.append(character)
                    chara_12vs45.append(character)
                    chara_1vs3vs5.append(character)
            else:
                nodule_no_fusions.append([patient_id,index,annotation])
        df_nodule_no_fusions=pd.DataFrame(nodule_no_fusions)
        df_nodule_no_fusions.to_csv('df_nodule_no_fusions.csv')
        print("nodule_pos",nodule_pos,"   ","nodule_neg",nodule_neg,"   ","nodule_3",nodule_3,"   ","used",nodule_3+nodule_neg+nodule_pos)
        return  y_1vs45, y_1vs5, y_12vs45, y_1vs3vs5, chara_1vs45, chara_1vs5, chara_12vs45, chara_1vs3vs5
    else:
        print("patiant",patient_id,"No nodules in annatations.csv")
        return None

if __name__ == "__main__":
    cubes_array = []
    cubes_1vs45 = []
    cubes_1vs5 = []
    cubes_12vs45 = []
    cubes_1vs3vs5 = []

    y_1vs45 = []
    y_1vs5 = []
    y_12vs45 = []
    y_1vs3vs5 = []

    chara_1vs45 = []
    chara_1vs5 = []
    chara_12vs45 = []
    chara_1vs3vs5 = []

    cubes_1vs45_pos = 0
    cubes_1vs5_pos = 0
    cubes_12vs45_pos = 0
    cubes_1vs3vs5_pos = 0
    cubes_1vs3vs5_3 = 0

    y_1vs45_pos = 0
    y_1vs5_pos = 0
    y_12vs45_pos = 0
    y_1vs3vs5_pos = 0
    y_1vs3vs5_3 = 0

    cubes_1vs45_neg = 0
    cubes_1vs5_neg = 0
    cubes_12vs45_neg = 0
    cubes_1vs3vs5_neg = 0

    y_1vs45_neg = 0
    y_1vs5_neg = 0
    y_12vs45_neg = 0
    y_1vs3vs5_neg = 0

    chara_1vs45_pos = 0
    chara_1vs5_pos = 0
    chara_12vs45_pos = 0
    chara_1vs3vs5_pos = 0

    candidate_index = 0
    only_patient = "197063290812663596858124411210"
    only_patient = None

    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir=settings.LUNA16_RAW_SRC_DIR+"subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if only_patient is not None and only_patient not in src_path:
                continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print(candidate_index, " patient: ", patient_id)
            process_pos_annotations_patient(src_path, patient_id)
            candidate_index += 1
            print("cubes_1vs5:  ","pos",cubes_1vs5_pos,"neg",cubes_1vs5_neg,"cube_length",len(cubes_1vs5),"    ","y_1vs5","pos",y_1vs5_pos,"neg",y_1vs5_neg,"y_length",len(y_1vs5))
            print("cubes_1vs45:  ", "pos", cubes_1vs45_pos, "neg", cubes_1vs45_neg, "cube_length", len(cubes_1vs45), "    ", "y_1vs45", "pos", y_1vs45_pos, "neg", y_1vs45_neg, "y_length", len(y_1vs45))
            print("cubes_12vs45:  ", "pos", cubes_12vs45_pos, "neg", cubes_12vs45_neg, "cube_length",len(cubes_12vs45), "    ", "y_12vs45", "pos", y_12vs45_pos, "neg", y_12vs45_neg, "y_length", len(y_12vs45))
            print("cubes_1vs3vs5:  ", "pos", cubes_1vs3vs5_pos,"suspic",cubes_1vs3vs5_3 , "neg", cubes_1vs3vs5_neg, "cube_length", len(cubes_1vs3vs5), "    ", "y_1vs3vs5","pos", y_1vs3vs5_pos,"suspic",y_1vs3vs5_3, "neg", y_1vs3vs5_neg, "y_length", len(y_1vs3vs5))

    cubes_1vs45=numpy.array(cubes_1vs45).reshape(numpy.array(cubes_1vs45).shape+(1,))
    cubes_12vs45=numpy.array(cubes_12vs45).reshape(numpy.array(cubes_12vs45).shape+(1,))
    cubes_1vs5=numpy.array(cubes_1vs5).reshape(numpy.array(cubes_1vs5).shape+(1,))
    cubes_1vs3vs5=numpy.array(cubes_1vs3vs5).reshape(numpy.array(cubes_1vs3vs5).shape+(1,))
    y_1vs45=numpy.array(y_1vs45)
    y_12vs45=numpy.array(y_12vs45)
    y_1vs5=numpy.array(y_1vs5)
    y_1vs3vs5=numpy.array(y_1vs3vs5)
    chara_1vs45=numpy.array(chara_1vs45)
    chara_12vs45=numpy.array(chara_12vs45)
    chara_1vs5=numpy.array(chara_1vs5)
    chara_1vs3vs5=numpy.array(chara_1vs3vs5)

    print("cubes_1vs45_shape",cubes_1vs45.shape)
    print("cubes_1vs5_shape",cubes_1vs5.shape)
    print("cubes_12vs45_shape",cubes_12vs45.shape)
    print("cubes_1vs3vs5_shape",cubes_1vs3vs5.shape)
    print("y_1vs45_shape",y_1vs45.shape)
    print("y_1vs5_shape",y_1vs5.shape)
    print("y_12vs45_shape",y_12vs45.shape)
    print("y_1vs3vs5_shape",y_1vs3vs5.shape)
    print("chara_1vs45_shape",chara_1vs45.shape)
    print("chara_1vs5_shape",chara_1vs5.shape)
    print("chara_12vs45_shape",chara_12vs45.shape)
    print("chara_1vs3vs5_shape",chara_1vs3vs5.shape)


    size=32
    numpy.save("data/%s/cubes_1vs45.npy"%size,cubes_1vs45)
    numpy.save("data/%s/cubes_12vs45.npy"%size,cubes_12vs45)
    numpy.save("data/%s/cubes_1vs5.npy"%size,cubes_1vs5)
    numpy.save("data/%s/cubes_1vs3vs5.npy"%size,cubes_1vs3vs5)
    numpy.save("data/%s/y_1vs45.npy"%size,y_1vs45)
    numpy.save("data/%s/y_12vs45.npy"%size,y_12vs45)
    numpy.save("data/%s/y_1vs5.npy"%size,y_1vs5)
    numpy.save("data/%s/y_1vs3vs5.npy"%size,y_1vs3vs5)
    numpy.save("data/%s/chara_1vs45.npy"%size,chara_1vs45)
    numpy.save("data/%s/chara_12vs45.npy"%size,chara_12vs45)
    numpy.save("data/%s/chara_1vs5.npy"%size,chara_1vs5)
    numpy.save("data/%s/chara_1vs3vs5.npy"%size,chara_1vs3vs5)



