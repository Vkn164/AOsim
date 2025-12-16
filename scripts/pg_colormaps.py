import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

def mpl_cmap_to_pg(name, n=256):
    cmap = plt.get_cmap(name)
    colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.ubyte)
    pos = np.linspace(0, 1, n)
    return pg.ColorMap(pos, colors)

def apply_mpl_cmap(image_item, img, cmap="viridis", vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)

    image_item.setImage(img, levels=(vmin, vmax), autoLevels=False)
    pg_cmap = mpl_cmap_to_pg(cmap)
    image_item.setLookupTable(pg_cmap.getLookupTable())
