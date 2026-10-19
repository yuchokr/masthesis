#!/usr/bin/env python3
from pathlib import Path
import pickle, warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import polars as pl
import geopandas as gpd
import spatialdata as sd
import xarray as xr
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
import ovrlpy

PROJECT_DIR = Path('/data/gent/vo/000/gvo00070/vsc48277/yujin_project')
ZARR_PATH = PROJECT_DIR / '40k_subset_0.zarr'
MAXPROJ_LABEL_LAYER = 'segmentation_mask_optimized_maxproj_45k'
STITCH_OUTPUT_DIR = PROJECT_DIR / '40k_subset_0_cellpose_fullcell_stitch3d_tiled_filtered'
INTENSITY_CSV = STITCH_OUTPUT_DIR / 'dapi_maxproj_mask_z_intensity_profiles' / 'stack_z_intensity_maxproj_mask_dapi.csv'
OVRLPY_CACHE = PROJECT_DIR / '40k_subset_0_cellpose_preprocessed' / 'ovrlpy_cache.pkl'
OUTPUT_DIR = STITCH_OUTPUT_DIR / 'ovrlpy_vsi_maxproj_masks'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
POLYGON_FILE = OUTPUT_DIR / 'stack_maxproj_polygons.parquet'
RAW_VSI_FILE = OUTPUT_DIR / 'stack_vsi_pixels.parquet'
SUMMARY_CSV = OUTPUT_DIR / 'stack_vsi_summary_maxproj_mask.csv'

PIXEL_SIZE_X_UM = 0.108
PIXEL_SIZE_Y_UM = 0.108
X_ORIGIN_UM = -40.0
Y_ORIGIN_UM = -110.0
Y_AXIS_FLIPPED = False
MIN_SIGNAL = 1.5
LOW_VSI_THRESHOLD = 0.7


def resolve_labels2d_level0(layer):
    print('\nInspecting SpatialData label structure...')
    print('Layer type:', type(layer))

    if isinstance(layer, xr.DataArray):
        da = layer
    else:
        da = None
        try:
            keys = list(layer.keys())
            print('Layer keys:', keys)
        except Exception:
            keys = []

        if 'scale0' in keys:
            scale0 = layer['scale0']
            print('scale0 type:', type(scale0))

            if isinstance(scale0, xr.DataArray):
                da = scale0
            elif hasattr(scale0, 'image') and isinstance(scale0.image, xr.DataArray):
                da = scale0.image
            else:
                try:
                    candidate = scale0['image']
                    if isinstance(candidate, xr.DataArray):
                        da = candidate
                except Exception:
                    pass

            if da is None and isinstance(scale0, xr.Dataset) and len(scale0.data_vars) == 1:
                da = next(iter(scale0.data_vars.values()))

        if da is None and hasattr(layer, 'image') and isinstance(layer.image, xr.DataArray):
            da = layer.image

    if da is None:
        raise TypeError('Could not resolve max-projection labels element to xarray.DataArray.')

    print('\nResolved label before squeeze:')
    print('  dims :', da.dims)
    print('  shape:', da.shape)
    print('  ndim :', da.ndim)

    da = da.squeeze(drop=True)

    print('\nResolved label after squeeze:')
    print('  dims :', da.dims)
    print('  shape:', da.shape)
    print('  ndim :', da.ndim)

    if da.ndim != 2:
        raise ValueError(f'Expected 2D maximum-projection labels, got shape={da.shape}, dims={da.dims}.')
    if set(da.dims) != {'y', 'x'}:
        raise ValueError(f"Expected dims ('y','x'), got {da.dims}.")

    da = da.transpose('y', 'x')
    print('\nFinal max-projection label:')
    print('  dims :', da.dims)
    print('  shape:', da.shape)
    print('  dtype:', da.dtype)
    return da


def maxproj_cell_to_polygon(maxproj_da, cell_id, y_min, y_max, x_min, x_max, full_transform):
    cell_id = int(cell_id)
    y_min, y_max = int(y_min), int(y_max)
    x_min, x_max = int(x_min), int(x_max)

    if y_max <= y_min or x_max <= x_min:
        print(f'[WARNING] Invalid ROI for cell {cell_id}: y=({y_min},{y_max}), x=({x_min},{x_max})')
        return None

    height = int(maxproj_da.sizes['y'])
    width = int(maxproj_da.sizes['x'])
    y_min = max(0, y_min); x_min = max(0, x_min)
    y_max = min(height, y_max); x_max = min(width, x_max)
    if y_max <= y_min or x_max <= x_min:
        return None

    crop_da = maxproj_da.isel(y=slice(y_min, y_max), x=slice(x_min, x_max))
    crop = np.asarray(crop_da.compute().to_numpy())

    if crop.ndim != 2:
        raise ValueError(
            f'Expected a 2D maxproj crop for cell_id={cell_id}, got shape={crop.shape}, dims={crop_da.dims}.'
        )

    binary_mask = (crop == cell_id).astype(np.uint8, copy=False)
    if not np.any(binary_mask):
        print(f'[WARNING] maxproj_cell_id={cell_id} not found inside saved ROI y=({y_min},{y_max}), x=({x_min},{x_max})')
        return None

    crop_transform = full_transform * Affine.translation(x_min, y_min)
    parts = []

    for geometry_mapping, value in shapes(
        source=binary_mask,
        mask=(binary_mask == 1),
        transform=crop_transform,
        connectivity=8,
    ):
        if int(value) != 1:
            continue
        geom = shape(geometry_mapping)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty:
            parts.append(geom)

    if not parts:
        return None

    geom = parts[0] if len(parts) == 1 else unary_union(parts)
    if not geom.is_valid:
        geom = geom.buffer(0)
    return None if geom.is_empty else geom


print('\n[1/9] Loading stack -> maxproj matching metadata...')
if not INTENSITY_CSV.exists():
    raise FileNotFoundError(f'Intensity CSV not found:\n{INTENSITY_CSV}')

intensity_df = pd.read_csv(INTENSITY_CSV)
required = {'stack_id','maxproj_cell_id','roi_y_min','roi_y_max','roi_x_min','roi_x_max'}
missing = required - set(intensity_df.columns)
if missing:
    raise ValueError(f'Missing required columns: {sorted(missing)}')

stack_roi_df = (
    intensity_df[['stack_id','maxproj_cell_id','roi_y_min','roi_y_max','roi_x_min','roi_x_max']]
    .dropna()
    .drop_duplicates(subset=['stack_id'])
    .copy()
)
for c in stack_roi_df.columns:
    stack_roi_df[c] = stack_roi_df[c].astype(np.int64)

print(f'Unique stacks:        {len(stack_roi_df):,}')
print(f'Unique maxproj cells: {stack_roi_df.maxproj_cell_id.nunique():,}')

print('\n[2/9] Loading max-projection Labels2D layer...')
sdata = sd.read_zarr(str(ZARR_PATH))
if MAXPROJ_LABEL_LAYER not in sdata.labels:
    raise KeyError(f"Label layer '{MAXPROJ_LABEL_LAYER}' not found. Available: {list(sdata.labels.keys())}")
maxproj_da = resolve_labels2d_level0(sdata.labels[MAXPROJ_LABEL_LAYER])

print('\n[3/9] Creating pixel-to-global transform...')
y_scale = -PIXEL_SIZE_Y_UM if Y_AXIS_FLIPPED else PIXEL_SIZE_Y_UM
pixel_to_global = Affine(PIXEL_SIZE_X_UM,0.0,X_ORIGIN_UM,0.0,y_scale,Y_ORIGIN_UM)
print(pixel_to_global)

print('\n[4/9] Converting matched maxproj masks to polygons...')
polygon_cache, polygon_rows = {}, []
n_total = len(stack_roi_df)
n_missing = 0

for i, row in enumerate(stack_roi_df.itertuples(index=False), start=1):
    stack_id = int(row.stack_id)
    maxproj_cell_id = int(row.maxproj_cell_id)
    key = (maxproj_cell_id,int(row.roi_y_min),int(row.roi_y_max),int(row.roi_x_min),int(row.roi_x_max))

    if key in polygon_cache:
        geom = polygon_cache[key]
    else:
        geom = maxproj_cell_to_polygon(
            maxproj_da=maxproj_da,
            cell_id=maxproj_cell_id,
            y_min=row.roi_y_min,
            y_max=row.roi_y_max,
            x_min=row.roi_x_min,
            x_max=row.roi_x_max,
            full_transform=pixel_to_global,
        )
        polygon_cache[key] = geom

    if geom is None:
        n_missing += 1
        continue

    polygon_rows.append({
        'stack_id': stack_id,
        'maxproj_cell_id': maxproj_cell_id,
        'roi_y_min': int(row.roi_y_min),
        'roi_y_max': int(row.roi_y_max),
        'roi_x_min': int(row.roi_x_min),
        'roi_x_max': int(row.roi_x_max),
        'geometry': geom,
    })

    if i % 5000 == 0 or i == n_total:
        print(f'  processed {i:,}/{n_total:,}')

if not polygon_rows:
    raise RuntimeError('No valid polygons were created.')

stack_polygons = gpd.GeoDataFrame(polygon_rows, geometry='geometry', crs=None)
print(f'Stacks with valid polygons: {len(stack_polygons):,}')
print(f'Stacks skipped:             {n_missing:,}')

bounds = stack_polygons.total_bounds
print(f'Polygon bounds: x=[{bounds[0]:.3f}, {bounds[2]:.3f}], y=[{bounds[1]:.3f}, {bounds[3]:.3f}]')
stack_polygons.to_parquet(POLYGON_FILE, index=False)
print(f'Saved polygons:\n{POLYGON_FILE}')

print('\n[5/9] Loading ovrlpy object...')
if not OVRLPY_CACHE.exists():
    raise FileNotFoundError(f'ovrlpy cache not found:\n{OVRLPY_CACHE}')
with open(OVRLPY_CACHE, 'rb') as f:
    liver = pickle.load(f)
print('Loaded object type:', type(liver))

print('\n[6/9] Calculating VSI pixels with ovrlpy...')
masks = stack_polygons['geometry']
ids = stack_polygons['stack_id'].astype(np.int64)
print(f'Number of masks: {len(masks):,}')
cell_pixels = ovrlpy.cell_integrity_from_masks(liver, masks, ids)
print(f'Returned VSI-pixel rows: {cell_pixels.height:,}')
cell_pixels.write_parquet(RAW_VSI_FILE)
print(f'Saved raw VSI pixels:\n{RAW_VSI_FILE}')

print('\n[7/9] Filtering low-signal pixels...')
filtered_pixels = cell_pixels.filter(pl.col('signal') > MIN_SIGNAL)
print(f'Before filtering: {cell_pixels.height:,}')
print(f'After signal > {MIN_SIGNAL}: {filtered_pixels.height:,}')

print('\n[8/9] Aggregating VSI per stack...')
filtered_pixels = filtered_pixels.with_columns(pl.col('cell_id').cast(pl.Int64).alias('stack_id'))

vsi_summary = (
    filtered_pixels
    .group_by('stack_id')
    .agg([
        pl.col('vsi').mean().alias('mean_vsi'),
        pl.col('vsi').median().alias('median_vsi'),
        pl.col('vsi').std().alias('std_vsi'),
        pl.col('vsi').min().alias('min_vsi'),
        pl.col('vsi').max().alias('max_vsi'),
        pl.col('signal').mean().alias('mean_signal'),
        pl.len().alias('n_vsi_pixels'),
        (((pl.col('vsi') < LOW_VSI_THRESHOLD).sum()) / pl.len()).alias('fraction_low_vsi_pixels'),
    ])
    .sort('stack_id')
)

print('\n[9/9] Saving final stack-level VSI table...')
metadata_pd = (
    stack_polygons[['stack_id','maxproj_cell_id','roi_y_min','roi_y_max','roi_x_min','roi_x_max']]
    .drop_duplicates(subset=['stack_id'])
    .copy()
)
metadata_pl = pl.from_pandas(metadata_pd)

final_vsi = (
    metadata_pl
    .join(vsi_summary, on='stack_id', how='left')
    .with_columns([
        pl.lit(MIN_SIGNAL).alias('minimum_signal_threshold'),
        pl.lit(LOW_VSI_THRESHOLD).alias('low_vsi_threshold'),
        pl.lit('matched_maximum_projection_segmentation_mask').alias('mask_definition'),
    ])
    .sort('stack_id')
)

final_vsi.write_csv(SUMMARY_CSV)
n_with_vsi = final_vsi.filter(pl.col('mean_vsi').is_not_null()).height
print('\nDone.')
print(f'Total matched stacks: {final_vsi.height:,}')
print(f'Stacks with VSI:      {n_with_vsi:,}')
print(f'Final summary:\n{SUMMARY_CSV}')
