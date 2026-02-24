import sparrow as sp
from spatialdata import read_zarr

zarr_path = "larger_subset.zarr"
sdata = read_zarr(zarr_path)

img_layer = list(sdata.images.keys())[0]
print("Input image layer:", img_layer)

# min-max filtering
sdata = sp.im.min_max_filtering(
    sdata=sdata,
    img_layer=img_layer,
    output_layer="min_max_filtered",
    size_min_max_filter=80,
    overwrite=True,
)

# CLAHE
sdata = sp.im.enhance_contrast(
    sdata=sdata,
    img_layer="min_max_filtered",
    output_layer="clahe",
    contrast_clip=15.0,
    chunks=20000,
    overwrite=True,
)

# plot all z layers (one figure per z)
z_values = [float(z) for z in sdata.images[img_layer].coords["z"].values]
for z_vis in z_values:
    sp.pl.plot_shapes(
        sdata,
        img_layer=[img_layer, "min_max_filtered", "clahe"],
        shapes_layer=None,
        z_slice=z_vis,
        channel="DAPI",
        figsize=(18, 6),
        img_title=True,
    )
