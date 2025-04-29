import tifffile, os, json, csv
import numpy as np
import pandas as pd
from math import ceil,floor
from skimage import measure

def label_coords(binary_image):
    labeled_img = measure.label(binary_image, connectivity=1)
    properties = measure.regionprops(labeled_img)
    centroid_list = []
    coords_list = []
    for pro in properties:
        centroid = pro.centroid
        coord = pro.coords
        centroid_list.append(centroid)
        coords_list.append(coord)

    return centroid_list, coords_list

def detection_gene(coords_lists, gene_image):
    region_pixels = gene_image[[coord[0] for coord in coords_lists], [coord[1] for coord in coords_lists], [coord[2] for coord in coords_lists]]
    max_intensity = np.amax(region_pixels)
    sum_intensity = np.sum(region_pixels)
    if max_intensity == 255 and sum_intensity >= 255:
        return 'True'
    else:
        return 'False'

def Gene_counts(dapi_root, Gene1_root, Gene2_root, save_root, z_index, image_id):
    cell = {}
    cell['ID'] = {}
    cell_id = 0
    image_dapi = tifffile.imread(os.path.join(dapi_root, 'Z{:05d}_image.tif'.format(image_id)))
    gene1_image = tifffile.imread(os.path.join(Gene1_root, 'Z{:05d}_image.tif'.format(image_id)))
    gene2_image = tifffile.imread(os.path.join(Gene2_root, 'Z{:05d}_image.tif'.format(image_id)))

    dapi_centroid_list, coords_list = label_coords(image_dapi)

    for d in range(len(dapi_centroid_list)):
        cell_id += 1
        cell['ID'][str(cell_id)] = {}
        cell['ID'][str(cell_id)]['vglut1'] = 'False'
        cell['ID'][str(cell_id)]['vgat'] = 'False'
        cell['ID'][str(cell_id)]['Index (x, y, z)'] = [ceil(dapi_centroid_list[d][2]), ceil(dapi_centroid_list[d][1]), ceil(z_index + dapi_centroid_list[d][0])]
        cell['ID'][str(cell_id)]['Physical coordinate'] = [ceil(4*dapi_centroid_list[d][2]), ceil(4*dapi_centroid_list[d][1]), ceil(4*(z_index + dapi_centroid_list[d][0]))]
    
        cell['ID'][str(cell_id)]['vglut1'] = detection_gene(coords_list[d], gene1_image)
        cell['ID'][str(cell_id)]['vgat'] = detection_gene(coords_list[d], gene2_image)
       

    with open(os.path.join(save_root, 'celltype_detection', 'Fish_vglut1_vgat_{}.json'.format(image_id)), 'w') as f:
         json.dump(cell, f, indent=4) 

    print('Finished')

def json2csv(save_root, name, gene_name='vGlut1'):
    os.makedirs(save_root, exist_ok=True)
    gene_x_list = []
    gene_y_list = []
    gene_z_list = []
    id = [55, 56, 57, 58, 59, 60, 61]
    for i in id:
        json_path = r'S:\Weijie\20250122_WJY_MHW_THREEGENES-NO2_B2_1\celltype_detection\Fish_vglut1_vgat_{}.json'.format(i)
        with open(json_path) as f:
            fish_json = json.load(f)
            for id in range(1, len(fish_json['ID'])+1):
                if gene_name == 'vGlut1':
                    if fish_json['ID'][str(id)]['vglut1'] == 'True' and fish_json['ID'][str(id)]['vgat'] == 'False':
                    # if fish_json['ID'][str(id)]['vglut1'] == 'True':
                        # if 2735 <= fish_json['ID'][str(id)]['Index (x, y, z)'][0] <= 2935 and 1380 <= fish_json['ID'][str(id)]['Index (x, y, z)'][1] <= 1580:
                            gene_x_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][0])
                            gene_y_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][1])
                            gene_z_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][2])
                if gene_name == 'vGat':
                    if fish_json['ID'][str(id)]['vglut1'] == 'False' and fish_json['ID'][str(id)]['vgat'] == 'True':
                    # if fish_json['ID'][str(id)]['vgat'] == 'True':
                        # if 2735 <= fish_json['ID'][str(id)]['Index (x, y, z)'][0] <= 2935 and 1380 <= fish_json['ID'][str(id)]['Index (x, y, z)'][1] <= 1580:
                            gene_x_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][0])
                            gene_y_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][1])
                            gene_z_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][2])
                if gene_name == 'vGlut1_vGat':
                    if fish_json['ID'][str(id)]['vglut1'] == 'True' and fish_json['ID'][str(id)]['vgat'] == 'True':
                        # if 2735 <= fish_json['ID'][str(id)]['Index (x, y, z)'][0] <= 2935 and 1380 <= fish_json['ID'][str(id)]['Index (x, y, z)'][1] <= 1580:
                            gene_x_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][0])
                            gene_y_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][1])
                            gene_z_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][2])
                if gene_name == "DAPI":
                    # if 2735 <= fish_json['ID'][str(id)]['Index (x, y, z)'][0] <= 2935 and 1380 <= fish_json['ID'][str(id)]['Index (x, y, z)'][1] <= 1580:
                        gene_x_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][0])
                        gene_y_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][1])
                        gene_z_list.append(fish_json['ID'][str(id)]['Index (x, y, z)'][2])
  
    save_name = os.path.join(save_root, 'Gene_' + gene_name + '_coordinate_'+ str(name) +'.csv')
    csv = pd.DataFrame(columns=['Coordinate_x', 'Coordinate_y'])
    csv['Coordinate_x'] = gene_x_list
    csv['Coordinate_y'] = gene_y_list
    csv['Coordinate_z'] = gene_z_list
    csv.to_csv(save_name, index=False)
    print('Finished')


if __name__ == '__main__':

    root = r'R:\WeijieZheng\model_infer'
    gene1_root = os.path.join(root, 'whole_brain_pred_dapi')
    gene2_root = os.path.join(root, 'whole_brain_pred_vglut1')
    gene3_root = os.path.join(root, 'whole_brain_pred_vgat')

    id = [55, 56, 57, 58, 59, 60, 61]
    for i in id:
        Gene_counts(gene1_root, gene2_root, gene3_root, save_root=root, z_index=(i-1)*64, image_id=i)
    save_root = ''
    json2csv(save_root=save_root, name='', gene_name='vGlut1')