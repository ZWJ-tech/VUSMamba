import os
import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.colors import TwoSlopeNorm
from scipy import spatial
from sklego.mixture import GMMClassifier
from scipy.ndimage import gaussian_filter
from PIL import Image
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline, UnivariateSpline

def relative_expression(csv_path, gene1, gene2, radius, save_path): 
    """
    This is to compute the relative spatial expression relationship of two genes, gene1 and gene2. 
    
    roi: pandas dataframe containing neuron id and their x,y,z positions (in the first 3 columns)
    spotcount: pandas dataframe containing neuron id and their gene expression (spot count)
    gene1 and gene2: genes to compute the relative relationship
    radius: the neighborhood radius used to compute the spatial gene expression 

    """   
    gene1=str(gene1)
    gene2=str(gene2)
    roi = pd.read_csv(os.path.join(csv_path, 'Gene_coordinate_no-overlap.csv'))
    X=roi.to_numpy()[:,:3] 
    neuron=spatial.KDTree(X)
    neighbors=neuron.query_ball_point(X,radius)
    roi['cluster']=roi['cluster'].astype(str)
    roi['fraction_%s_%s' % (str(gene1), str(gene2))]=0

    ind1=roi.columns.get_loc('cluster')
    ind2=roi.columns.get_loc('fraction_%s_%s' % (str(gene1), str(gene2)))
    for i in range(0,len(neighbors)):
        x=[]
        for j in neighbors[i]:
            x=np.append(x, roi.iloc[j,ind1])
        a,b=np.unique(x,return_counts=True)
        if np.any(a==gene1) and np.any(a==gene2):
            c=b[np.argwhere(a==gene1)]
            d=b[np.argwhere(a==gene2)]
            roi.iloc[i,ind2]=float((c-d)/(c+d))
        else:
            if np.any(a==gene1):
                roi.iloc[i,ind2]=1
            if np.any(a==gene2):
                roi.iloc[i,ind2]=-1

    for i in roi.index:
        if roi.loc[i, 'fraction_%s_%s' % (str(gene1), str(gene2))]> 0:
            roi.loc[i,'classifier']='1' 
        else:
            roi.loc[i,'classifier']='2'  

    save = os.path.join(save_path, 'result.csv')
    roi.to_csv(save, index=False)
    return 'Finished'

def plot_relative_expression(csv_path, column, num_z, invert_x=False, invert_y=False, invert_z=False):
    """
    Plot the relative spatial expression relationship of two genes, as computed with the 'relative_expression' function. 
    
    roi: pandas dataframe containing neuron id and their x,y,z positions (in the first 3 columns)
    column: string, the column name containing the relative expression 
    num_z: number of axial levels to plot.   
    invert_x, invert_y, invert_z: True or False, whether to invert the x,y and z axis. Default is False
    
    """   
    roi = pd.read_csv(os.path.join(csv_path, 'result.csv'))
    width_ratio=np.ones(num_z)
    width_ratio=np.append(width_ratio,0.1)               
    fig,ax=plt.subplots(1,num_z+1,figsize=(num_z*5+1,5),dpi=150,gridspec_kw={"width_ratios":width_ratio})
    ind=0
    A=roi.copy()
    z_min = A.z.min() 
    z_max = A.z.max()  
    x_min = A.x.min() 
    x_max = A.x.max()  
    y_min = A.y.min() 
    y_max = A.y.max()  
    s = (z_max - z_min) / num_z 
    column=str(column)
    if invert_z:
        A.z=A.z.max()-A.z
    for n in range(1,num_z+1):
        # B=A[(A.z>((n-1)*s))&(A.z<=(n*s))]
        layer_start = z_min + (n-1)*s
        layer_end = z_min + n*s
        B = A[(A.z > layer_start) & (A.z <= layer_end)]
        x=roi[roi.index.isin(B.index)][column].astype(float)
        # a=ax.flatten()[ind].scatter(B.to_numpy()[:,2],(B.to_numpy()[:,1]), c=x, 
        #                             norm=TwoSlopeNorm(vcenter=0),
        #                             marker='o', s=10, cmap=plt.cm.coolwarm,alpha=1)
        a = ax.flatten()[ind].scatter(
            B.iloc[:, 0],  
            B.iloc[:, 1],  
            c=x, 
            norm=TwoSlopeNorm(vcenter=0),
            marker='o', 
            s=10, 
            cmap=plt.cm.coolwarm,
            alpha=1
        )                            
        ax.flatten()[ind].yaxis.set_tick_params(pad=10,labelsize=12)
        ax.flatten()[ind].xaxis.set_tick_params(pad=3,labelsize=12)
        ax.flatten()[ind].set_xlim(x_min,x_max)
        ax.flatten()[ind].set_ylim(y_min,y_max)
        ax.flatten()[ind].set(adjustable='box', aspect='equal')
        # ax.flatten()[ind].set_title(str(s*(n-1))+'-'+str(s*n)+'µm (z axis)',fontsize=16)
        ax.flatten()[ind].set_title(f"{layer_start:.1f}–{layer_end:.1f} µm (z)", fontsize=12)
        ax.flatten()[ind].xaxis.set_ticklabels([])
        ax.flatten()[ind].yaxis.set_ticklabels([])
        ax.flatten()[ind].xaxis.set_ticks([])
        ax.flatten()[ind].yaxis.set_ticks([])
        if invert_y: 
            ax.flatten()[ind].invert_yaxis()
        if invert_x:
            ax.flatten()[ind].invert_xaxis()
        ind = ind + 1
    cb=fig.colorbar(a, cax=ax.flatten()[ind], shrink=0.1,aspect=10,pad=0.5)
    cb.ax.set_title(column,size=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(csv_path, "output.pdf"), dpi=300, bbox_inches='tight', transparent=True)

def GMM_segmentation(csv_path):

    roi = pd.read_csv(os.path.join(csv_path, 'result.csv'))
    fig,ax=plt.subplots(2,4,figsize=(20,10),dpi=300)
    ind=0
    x_min, y_min, z_min = roi.x.min(), roi.y.min(), roi.z.min()
    x_max, y_max, z_max = roi.x.max(), roi.y.max(), roi.z.max()
    A=roi.copy()
    A=A[(A.x<x_max)&(A.y<y_max)&(A.z<z_max)]
    # A.z=A.z.max()-A.z
    boundaries = []
    mod = GMMClassifier(n_components=10).fit(A.to_numpy()[:,0:3], A['classifier'])

    for z in [3742, 3772, 3802, 3832]:
        B=A[(A.z>(z-30))&(A.z<=z)]
        a = np.linspace(x_min, x_max, 300)  
        b = np.linspace(y_min, y_max, 300)
        c= np.linspace(z-10, z-9, 1)   
        # xa, xb,xc = np.meshgrid(a, b, c) 
        z_val = z - 9.5 
        a_grid, b_grid = np.meshgrid(a, b)
        U = np.stack([a_grid.ravel(), b_grid.ravel(), np.full(a_grid.size, z_val)], axis=1)
        # U=np.zeros((xa.flatten().shape[0],3))
        # U[:,0] = xa.flatten()
        # U[:,1] = xb.flatten()
        # U[:,2] = xc.flatten()
        Z=mod.predict(U).astype(float)
        Z_grid = Z.reshape(a_grid.shape)
        Z_smoothed = gaussian_filter(Z_grid.astype(float), sigma=10)
        contours = ax[1, ind].contour(
            a_grid, b_grid, Z_smoothed, 
            levels=np.arange(0.5, 10.5, 1), 
            colors='k'
        )
        for level_segs in contours.allsegs:
            for seg in level_segs:
                if len(seg) > 0:
                    seg_3d = np.hstack([seg, np.full((len(seg), 1), z_val)])
                    seg_scaled = seg_3d / 12.5 
                    boundaries.append(seg_scaled)

        # a=ax[1,ind].scatter(U[:, 0], U[:, 1], c=Z, s=2,cmap=plt.cm.Paired)
        a=ax[1, ind].contourf(a_grid, b_grid, Z_grid, cmap=plt.cm.Paired, alpha=0.7, levels=100, antialiased=True)
        a=ax[0,ind].scatter(B.x, B.y, c=B.classifier.astype('float'), s=10,alpha=0.7,cmap=plt.cm.Paired)
        for mm in range(0,2):
            ax[mm,ind].xaxis.set_ticklabels([])
            ax[mm,ind].yaxis.set_ticklabels([])
            ax[mm,ind].xaxis.set_ticks([])
            ax[mm,ind].yaxis.set_ticks([])
            ax[mm,ind].set_ylim(y_min,y_max)
            ax[mm,ind].set_xlim(x_min,x_max)
            ax[mm,ind].invert_yaxis()
        ind+=1
    if boundaries:
        all_points = []
        for seg in boundaries:
            all_points.extend(seg.tolist())
            all_points.append([np.nan, np.nan, np.nan])  # 用NaN分隔线段
        
        df = pd.DataFrame(all_points, columns=['x', 'y', 'z'])
        df.to_csv(os.path.join(csv_path, 'boundaries.csv'), index=False)

    # plt.tight_layout()
    # plt.show()
    # plt.savefig(os.path.join(csv_path, "seg_contourf.pdf"), dpi=300, bbox_inches='tight', transparent=True)

def csv_to_tif(csv_path, save_path):
    csv = pd.read_csv(os.path.join(csv_path, 'boundaries_1.csv'))
    # x_list = csv['Coordinate_x'].to_list()
    # y_list = csv['Coordinate_y'].to_list()
    # z_list = csv['Coordinate_z'].to_list()
    x_list = csv['x'].to_list()
    y_list = csv['y'].to_list()
    z_list = csv['z'].to_list()
    region_gene = np.zeros((636, 400, 560))
    for i in range(len(x_list)):
        # x, y, z = ceil(x_list[i] // 12.5), ceil(y_list[i] // 12.5), ceil(z_list[i] // 12.5)
        x, y, z = ceil(x_list[i]), ceil(y_list[i]), ceil(z_list[i])
        region_gene[z, y, x] = 255

    tifffile.imwrite(os.path.join(save_path, 'boundaries_1' + '.tif'), region_gene.astype('uint16')) 
    print('Finished')

def plt_atlas_bounaries(path):

    image_name = 'atlas_25mic_305.jpg' 
    image = Image.open(os.path.join(path, image_name)).convert('L')
    plt.imshow(image, cmap='gray')
    plt.axis('off')  

    csv_paths = ['Gene_vGat_coordinate_305.csv', 'Gene_vGlut1_coordinate_305.csv', 'boundaries_4.csv']  
    rgb_colors = [
        (1.0, 0.0, 0.0),  
        (0.0, 1.0, 0.0),    
        (1.0, 1.0, 0.0)          
    ]

    for csv_path, color in zip(csv_paths, rgb_colors):
        df = pd.read_csv(os.path.join(path, csv_path))
        plt.scatter(df['x'], df['y'], c=[color], s=0.5, marker='o')  

    output_path = os.path.join(path, 'output_with_points_305.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

def curve_evl(csv_path):
    curve1 = pd.read_csv(os.path.join(csv_path, 'boundaries_4.csv'))  
    curve2 = pd.read_csv(os.path.join(csv_path, 'atlas_RT_edge_305.csv'))
    curve1 = curve1.sort_values('x').reset_index(drop=True)
    curve2 = curve2.sort_values('x').reset_index(drop=True)

    x_common = np.linspace(max(curve1['x'].min(), curve2['x'].min()),
                        min(curve1['x'].max(), curve2['x'].max()), 500)

    y1_interp = np.interp(x_common, curve1['x'], curve1['y'])
    y2_interp = np.interp(x_common, curve2['x'], curve2['y'])

    euclid = euclidean(y1_interp, y2_interp)

    mse = mean_squared_error(y1_interp, y2_interp)
    mae = mean_absolute_error(y1_interp, y2_interp)

    corr, _ = pearsonr(y1_interp, y2_interp)


    area_diff = np.trapz(np.abs(y1_interp - y2_interp), x_common)

    print(f"欧几里得距离: {euclid:.4f}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
    print(f"相关系数: {corr:.4f}")
    print(f"面积差异: {area_diff:.4f}")

def compute_normals(curve):
    tangents = np.zeros_like(curve)
    n_points = len(curve)
    
    for i in range(n_points):
        if i == 0:
            tangents[i] = curve[i+1] - curve[i]
        elif i == n_points - 1:
            tangents[i] = curve[i] - curve[i-1]
        else:
            tangents[i] = (curve[i+1] - curve[i-1]) / 2.0
    
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms[tangent_norms == 0] = 1e-12
    tangents_normalized = tangents / tangent_norms
    
    normals = np.empty_like(tangents_normalized)
    normals[:, 0] = -tangents_normalized[:, 1]
    normals[:, 1] = tangents_normalized[:, 0]
    
    return normals

def interpolate_curve(curve, num_points=1000):

    diff = np.diff(curve, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    t = np.insert(np.cumsum(dist), 0, 0)
    t_normalized = t / t[-1]

    cs_x = CubicSpline(t_normalized, curve[:, 0])
    cs_y = CubicSpline(t_normalized, curve[:, 1])

    # 生成新的参数化点
    t_new = np.linspace(0, 1, num_points)
    curve_interp = np.column_stack((cs_x(t_new), cs_y(t_new)))
    
    return curve_interp

def interpolate_curve_v2(curve, num_points=1000, method='smoothing', s=None):
    """支持平滑的曲线插值方法"""
    # 参数化处理（弦长参数化）
    diff = np.diff(curve, axis=0)
    dist = np.linalg.norm(diff, axis=1)
    t = np.insert(np.cumsum(dist), 0, 0)
    t_normalized = t / t[-1]

    # 选择插值方法
    if method == 'cubic':
        # 三次样条插值（精确通过点）
        cs_x = CubicSpline(t_normalized, curve[:, 0], bc_type='clamped')
        cs_y = CubicSpline(t_normalized, curve[:, 1], bc_type='clamped')
        t_new = np.linspace(0, 1, num_points)
        return np.column_stack((cs_x(t_new), cs_y(t_new)))
    
    elif method == 'smoothing':
        # 平滑样条拟合（允许误差）
        # 需要为UnivariateSpline指定权重或s参数
        if s is None:
            s = 0.1 * len(curve)  # 默认平滑因子
        
        # 分别拟合x和y分量
        spline_x = UnivariateSpline(t_normalized, curve[:, 0], s=s)
        spline_y = UnivariateSpline(t_normalized, curve[:, 1], s=s)
        
        # 生成均匀参数点
        t_new = np.linspace(0, 1, num_points)
        return np.column_stack((spline_x(t_new), spline_y(t_new)))
    
    else:
        raise ValueError("Method must be 'cubic' or 'smoothing'")

def centripetal_parameterization(curve, n_points):
    """向心参数化（减少尖角处的过冲）"""
    diff = np.diff(curve, axis=0)
    dist = np.linalg.norm(diff, axis=1)**0.5  # 向心参数化关键修改
    t = np.insert(np.cumsum(dist), 0, 0)
    return t / t[-1] * (n_points-1) / n_points

def main(csv_path):
    curve_a = pd.read_csv(os.path.join(csv_path, "boundaries_4.csv"))[["x", "y"]].values
    curve_b_raw = pd.read_csv(os.path.join(csv_path, "atlas_RT_edge_305.csv"))[["x", "y"]].values

    # curve_b = interpolate_curve(curve_b_raw, num_points=424)
    curve_b = interpolate_curve_v2(curve_b_raw, num_points=415, s=4)

    normals_b = compute_normals(curve_b)
    kd_tree = KDTree(curve_b)
    distances, indices = kd_tree.query(curve_a, k=1)
    
    signed_dists = [
        np.dot(curve_a[i] - curve_b[idx], normals_b[idx])
        for i, idx in enumerate(indices)
    ]
    
    pd.DataFrame(signed_dists, columns=["signed_distance"]).to_csv(
        os.path.join(csv_path, "signed_distances_305_v2.csv"), index=False
    )
    pd.DataFrame(curve_b, columns=["x", "y"]).to_csv(
        os.path.join(csv_path, "atlas_RT_edge_305_v2.csv"), index=False
    )

if __name__ == "__main__":
    csv_path = r'S:\Weijie\20250122_WJY_MHW_THREEGENES-NO1_B2_1\celltype_detection\csv_parcellation'
    # relative_expression(csv_path, gene1='vglut1', gene2='vgat', radius=25, save_path=csv_path)
    # plot_relative_expression(csv_path, 'fraction_vglut1_vgat', 4, invert_y=True)
    # GMM_segmentation(csv_path)
    # csv_to_tif(csv_path, csv_path)
    # plt_atlas_bounaries(csv_path)
    # curve_evl(csv_path)
    main(csv_path)