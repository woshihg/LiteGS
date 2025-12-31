import numpy as np
import numpy.typing as npt
import os
from plyfile import PlyData, PlyElement


def save_ply(path:str,xyz:npt.NDArray,scale:npt.NDArray,rot:npt.NDArray,sh_0:npt.NDArray,sh_rest:npt.NDArray,opacity:npt.NDArray,features:npt.NDArray=None):
    
    xyz=xyz.transpose(1,0)
    scale=scale.transpose(1,0)
    rot=rot.transpose(1,0)
    sh_0=sh_0.transpose(2,1,0)
    sh_rest=sh_rest.transpose(2,1,0)
    opacity=opacity.transpose(1,0)
    if features is not None:
        features=features.transpose(1,0)
    
    dirname=os.path.dirname(path)
    os.makedirs(dirname,exist_ok=True)

    def construct_list_of_attributes(sh_0,sh_rest,features):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(sh_0.shape[-1]*sh_0.shape[-2]):
            l.append('f_dc_{}'.format(i))
        for i in range(sh_rest.shape[-1]*sh_rest.shape[-2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        if features is not None:
            for i in range(features.shape[-1]):
                l.append('feature_{}'.format(i))
        return l
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(sh_0,sh_rest,features)]
    points_num=xyz.shape[0]
    assert(scale.shape[0]==points_num)
    assert(rot.shape[0]==points_num)
    assert(sh_0.shape[0]==points_num)
    assert(sh_rest.shape[0]==points_num)
    assert(opacity.shape[0]==points_num)
    if features is not None:
        assert(features.shape[0]==points_num)
    elements = np.empty(points_num, dtype=dtype_full)
    
    if features is not None:
        attributes = np.concatenate((xyz, np.zeros_like(xyz), sh_0.reshape(points_num,-1), sh_rest.reshape(points_num,-1), 
                                        opacity, scale, rot, features), axis=1)
    else:
        attributes = np.concatenate((xyz, np.zeros_like(xyz), sh_0.reshape(points_num,-1), sh_rest.reshape(points_num,-1), 
                                        opacity, scale, rot), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    return

def load_ply(path:str,sh_degree:int):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    sh_0 = np.zeros((xyz.shape[0], 3, 1))
    sh_0[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    sh_0[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    sh_0[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Infer sh_degree from the number of extra features
    # Total SH coeffs = (sh_degree + 1)^2
    # Extra SH coeffs (excluding DC) = (sh_degree + 1)^2 - 1
    # Total extra properties = 3 * ((sh_degree + 1)^2 - 1)
    num_extra_features = len(extra_f_names)
    inferred_sh_degree = int(np.sqrt(num_extra_features // 3 + 1) - 1)
    
    sh_rest = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        sh_rest[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    if num_extra_features > 0:
        sh_rest = sh_rest.reshape((sh_rest.shape[0], 3, (inferred_sh_degree + 1) ** 2 - 1))
    else:
        sh_rest = sh_rest.reshape((sh_rest.shape[0], 3, 0))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scale = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rot = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rot[:, idx] = np.asarray(plydata.elements[0][attr_name])
    
    feature_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("feature_")]
    feature_names = sorted(feature_names, key = lambda x: int(x.split('_')[-1]))
    if len(feature_names) > 0:
        features = np.zeros((xyz.shape[0], len(feature_names)))
        for idx, attr_name in enumerate(feature_names):
            features[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features = features.transpose(1,0)
    else:
        features = None

    xyz=xyz.transpose(1,0)
    scale=scale.transpose(1,0)
    rot=rot.transpose(1,0)
    sh_0=sh_0.transpose(2,1,0)
    sh_rest=sh_rest.transpose(2,1,0)
    opacity=opacity.transpose(1,0)

    return xyz,scale,rot,sh_0,sh_rest,opacity,inferred_sh_degree,features