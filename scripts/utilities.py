import aotools
import cupy as cp
import numpy as np

from pathlib import Path
import json

with open(Path(__file__).parent.parent / "config_default.json", "r") as f:
    config = json.load(f)

params = config.copy()

def set_params(new_params):
    global params
    params = new_params

class PhaseMap_tools:
    @staticmethod
    def generate_phase_map(grid_size=None, telescope_diameter=None, r0=None, L0=None, Vwind=None, altitude=None, random_seed=None):
        # read current params when values not provided
        if grid_size is None:
            grid_size = params.get("grid_size")
        if telescope_diameter is None:
            telescope_diameter = params.get("telescope_diameter")
        if r0 is None:
            r0 = params.get("r0")
        if L0 is None:
            L0 = params.get("L0")
        if random_seed is None:
            random_seed = params.get("random_seed")

        # create ohase screen
        phase_screen = aotools.turbulence.infinitephasescreen.PhaseScreenKolmogorov(grid_size, telescope_diameter/grid_size, r0, L0, random_seed=random_seed)
        phase_map = phase_screen.add_row

        # generate next rows based on wind vel and dt
        def advance_phase_map(frame_rate=500):
            shift_m = Vwind * 1/frame_rate
            shift_pix = shift_m / (telescope_diameter/grid_size)
            n_rows = int(np.round(abs(shift_pix)))
            for _ in range(max(1, n_rows-1)):
                phase_map()
            return phase_map()
        
        return phase_screen, advance_phase_map
    
    # generate influence map for an amplitude and spread factor sigma
    @staticmethod
    def make_influence(r0, c0, amp=1e-6, sigma=None, grid_size=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if sigma is None:
            sigma = max(1.0, grid_size * 0.05)
        yy, xx = cp.meshgrid(cp.arange(grid_size), cp.arange(grid_size), indexing='ij')
        # surface (meters) shape (Ngrid,Ngrid) centered at r0,c0
        rdist2 = (yy - r0)**2 + (xx - c0)**2
        surf = amp * cp.exp(-0.5 * rdist2 / (sigma**2))
        # mask outside pupil to 0
        return cp.asarray(surf)

    # get slopes and centroids of each sub pupil/aperture given a phase map
    def measure_slopes_from_phase(phase_map, xs, ys, sub_pupil_masks, pad=params.get("field_padding")):
        
        sub_imgs = phase_map[ys, xs] 

        # field -> PSF -> centroid 
        field = sub_pupil_masks * cp.exp(1j * sub_imgs)
        field_p = cp.pad(field, ((0,0),(pad,pad),(pad,pad)), mode='constant')

        F = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(field_p, axes=(1,2)), axes=(1,2)), axes=(1,2))
        I = cp.abs(F)**2
        I /= cp.sum(I, axis=(1,2), keepdims=True)

        h_p, w_p = I.shape[1], I.shape[2]
        x_coords = cp.arange(w_p, dtype=cp.float32)[None, None, :]
        y_coords = cp.arange(h_p, dtype=cp.float32)[None, :, None]
        cx = cp.sum(I * x_coords, axis=(1,2))
        cy = cp.sum(I * y_coords, axis=(1,2))
        center = cp.array([h_p/2.0, w_p/2.0])            # (cy_center, cx_center)
        centroids = cp.stack((cy, cx), axis=1)           # (n_sub,2)
        slopes = (centroids - center[None, :]).reshape(-1) 

        return slopes, centroids






class Pupil_tools:
    @staticmethod
    def generate_pupil(grid_size=None, telescope_center_obscuration=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if telescope_center_obscuration is None:
            telescope_center_obscuration = params.get("telescope_center_obscuration")

        outer = aotools.pupil.circle(grid_size/2, grid_size)
        inner = aotools.pupil.circle(grid_size/2 * telescope_center_obscuration, grid_size)
        return cp.asarray(outer - inner)

    # generate actuator positions on pupil
    @staticmethod
    def generate_actuators(pupil=None, actuators=None, grid_size=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size)

        act_centers = []
        for i in range(actuators):
            for j in range(actuators):
                r = int((i + 0.5) * grid_size/actuators)
                c = int((j + 0.5) * grid_size/actuators)
                if pupil[r, c] > 0:
                    act_centers.append([r, c])
        return cp.asarray(act_centers)

    # generate list of images that denote each actuator's influence (from 0 to 1) on the mirror
    @staticmethod
    def generate_actuator_influence_map(act_centers=None, pupil=None, actuators=None, poke_amplitude=None, grid_size=None):
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size)
        if grid_size is None:
            grid_size = params.get("grid_size")
        if actuators is None:
            actuators = params.get("actuators")
        if poke_amplitude is None:
            poke_amplitude = params.get("poke_amplitude")
        if act_centers is None:
            act_centers = Pupil_tools.generate_actuators(pupil=pupil, actuators=actuators, grid_size=grid_size)

        influence_maps = []
        for r0, c0 in act_centers:
            surf_pos = PhaseMap_tools.make_influence(r0, c0, amp=poke_amplitude, sigma = grid_size/actuators * 0.6, grid_size=grid_size)
            influence_maps.append(surf_pos / poke_amplitude * pupil)
        return influence_maps

    # generate number of sub apertures, pixel positions, pupil/phase_map indices, sub aperture width, slice indices of each sub aperture, and sub pupil image/map 
    @staticmethod
    def generate_sub_apertures(pupil=None, sub_apertures=None, grid_size=None):
        if grid_size is None:
            grid_size = params.get("grid_size")
        if sub_apertures is None:
            sub_apertures = params.get("sub_apertures")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil(grid_size=grid_size)

        sub_aps = aotools.wfs.findActiveSubaps(sub_apertures, pupil, 0.6)
        sub_aps = cp.asarray(sub_aps)
        active_sub_aps = sub_aps.shape[0]

        sub_aps_idx = (sub_aps/(grid_size/sub_apertures)).astype(int)

        sub_ap_width = grid_size/sub_apertures

        sub_slice = [(slice(i[0], i[0]+int(sub_ap_width)+1), slice(i[1], i[1]+int(sub_ap_width)+1)) for i in (sub_aps_idx.get()*sub_ap_width).astype(int)]

        sub_pupils = cp.asarray([pupil[i] for i in sub_slice])

        return active_sub_aps, sub_aps, sub_aps_idx, sub_ap_width, sub_slice, sub_pupils

class Analysis:
    # for each subaperture find the centroid and generate an image 
    @staticmethod
    def generate_subaperture_images(k, pupil=None, influence_maps=None, sub_aps_idx=None, sub_ap_width=None, sub_pupils=None, poke_amplitude=None, wfs_lambda=None, pad=None):
        if poke_amplitude is None:
            poke_amplitude = params.get("poke_amplitude")
        if wfs_lambda is None:
            wfs_lambda = params.get("wfs_lambda")
        if pad is None:
            pad = params.get("field_padding")
        if pupil is None or sub_pupils is None or influence_maps is None or sub_aps_idx is None:
            active_sub_aps, sub_aps, sub_aps_idx, sub_ap_width, sub_slice, sub_pupils = Pupil_tools.generate_sub_apertures()
            influence_maps = Pupil_tools.generate_actuator_influence_map()
            pupil = Pupil_tools.generate_pupil()
        if influence_maps is None:
            influence_maps = Pupil_tools.generate_actuator_influence_map(
                                            act_centers=None,
                                            pupil=pupil,
                                            actuators=params.get("actuators"),
                                            poke_amplitude=params.get("poke_amplitude"),
                                            grid_size=params.get("grid_size")
                                        )
        
        h = int(sub_ap_width)+1
        w = int(sub_ap_width)+1

        yy = cp.arange(h)[:,None]
        xx = cp.arange(w)[None,:]

        top_left = ((sub_aps_idx * sub_ap_width).astype(int))  # shape (n_subaps, 2) on CPU
        top_left = cp.asarray(top_left)

        ys = top_left[:,0,None,None] + yy[None,:,:]  
        xs = top_left[:,1,None,None] + xx[None,:,:]

        # pick the k-th actuator poke
        phase_map = poke_amplitude * (4.0 * cp.pi / wfs_lambda) * influence_maps[k] * pupil
        
        # extract subaperture phases
        sub_imgs = phase_map[ys, xs]
        
        # field -> PSF
        field = sub_pupils * cp.exp(1j * sub_imgs)
        field_p = cp.pad(field, ((0,0),(pad,pad),(pad,pad)), mode='constant')

        F = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(field_p, axes=(1,2)), axes=(1,2)), axes=(1,2))
        I = cp.abs(F)**2
        I /= cp.sum(I, axis=(1,2), keepdims=True)
        
        # centroids
        h_p, w_p = I.shape[1], I.shape[2]
        x_coords = cp.arange(w_p, dtype=cp.float32)[None, None, :]
        y_coords = cp.arange(h_p, dtype=cp.float32)[None, :, None]
        cx = cp.sum(I * x_coords, axis=(1,2))
        cy = cp.sum(I * y_coords, axis=(1,2))
        centroids = cp.stack((cy, cx), axis=1)   # (n_sub, 2)

        n_subaps = sub_aps_idx.shape[0]
        h_p, w_p = I.shape[1], I.shape[2]

        # Number of subapertures along y/x (assuming square grid)
        max_y = int(cp.max(sub_aps_idx[:,0])) + 1
        max_x = int(cp.max(sub_aps_idx[:,1])) + 1

        H = max_y * h_p
        W = max_x * w_p

        sensor = cp.zeros((H, W), dtype=I.dtype)

        iy = sub_aps_idx[:,0].astype(cp.int32)
        ix = sub_aps_idx[:,1].astype(cp.int32)

        # create grids for each subap image
        yy = cp.arange(h_p)[None, :, None] + iy[:, None, None] * h_p  # shape (n_subaps, h_p, 1)
        xx = cp.arange(w_p)[None, None, :] + ix[:, None, None] * w_p  # shape (n_subaps, 1, w_p)

        # broadcast to full 3D grid
        yy = cp.broadcast_to(yy, (n_subaps, h_p, w_p))
        xx = cp.broadcast_to(xx, (n_subaps, h_p, w_p))

        # restitch individual subaperture image into one
        sensor[yy.ravel(), xx.ravel()] = I.ravel()

        return centroids, sensor
    
    # create PSF given pupil and phase map
    @staticmethod
    def generate_science_image(pupil=None, phase_map=None, science_lambda=None, pad=None):
        if pad is None:
            pad = params.get("field_padding")
        if science_lambda is None:
            science_lambda = params.get("science_lambda")
        if pupil is None:
            pupil = Pupil_tools.generate_pupil()

        field = pupil * cp.exp(1j * phase_map)
        field_p = cp.pad(field,((pad,pad) , (pad,pad)), mode='constant')

        F = cp.fft.fftshift(cp.fft.fft2(field_p))
        I = cp.abs(F)**2

        F0 = cp.fft.fftshift(cp.fft.fft2(pupil))
        I0 = cp.abs(F0)**2

        return I,  I.max()/I0.max()
    
    @staticmethod
    def fwhm_radial(I):

        xp = cp.get_array_module(I)

        ny, nx = I.shape
        cy, cx = ny // 2, nx // 2

        y, x = xp.indices(I.shape)
        r = xp.sqrt((x - cx)**2 + (y - cy)**2)

        r_flat = r.ravel()
        I_flat = I.ravel()

        # radial bins (integer pixel radii)
        r_int = r_flat.astype(xp.int32)
        max_r = r_int.max() + 1

        # azimuthal average
        radial_sum = xp.bincount(r_int, weights=I_flat, minlength=max_r)
        radial_cnt = xp.bincount(r_int, minlength=max_r)
        radial_prof = radial_sum / radial_cnt

        radial_prof /= radial_prof[0]  # normalize to peak

        # find half-maximum
        half = 0.5
        idx = xp.where(radial_prof <= half)[0][0]

        # linear interpolation for sub-pixel accuracy
        r1, r2 = idx - 1, idx
        y1, y2 = radial_prof[r1], radial_prof[r2]
        r_half = r1 + (half - y1) / (y2 - y1)

        return 2.0 * r_half




if __name__ == "__main__":
    import argparse
    import inspect
    import sys

    classes = {
        name: cls
        for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if not name.startswith("_")
    }

    # Collect all methods of all classes
    functions = {}
    for cls_name, cls in classes.items():
        for name, func in inspect.getmembers(cls, inspect.isfunction):
            if func.__module__ == __name__ and not name.startswith("_"):
                # store with "ClassName.method" as key
                functions[f"{cls_name}.{name}"] = func

    # Create parser
    parser = argparse.ArgumentParser(description="Run any function from this file")
    parser.add_argument("--list-functions", action="store_true", help="List available functions")
    parser.add_argument("function", nargs="?", choices=functions.keys(), help="Function to run")

    # AO parameters with defaults from config
    parser.add_argument("--telescope_diameter", type=float, default=config.get("telescope_diameter"))
    parser.add_argument("--telescope_center_obscuration", type=float, default=config.get("telescope_center_obscuration"))
    parser.add_argument("--wfs_lambda", type=float, default=config.get("wfs_lambda"))
    parser.add_argument("--science_lambda", type=float, default=config.get("science_lambda"))
    parser.add_argument("--r0", type=float, default=config.get("r0"))
    parser.add_argument("--L0", type=float, default=config.get("L0"))
    parser.add_argument("--Vwind", type=float, default=config.get("Vwind"))
    parser.add_argument("--actuators", type=int, default=config.get("actuators"))
    parser.add_argument("--sub_apertures", type=int, default=config.get("sub_apertures"))
    parser.add_argument("--frame_rate", type=int, default=config.get("frame_rate"))
    parser.add_argument("--grid_size", type=int, default=config.get("grid_size"))
    parser.add_argument("--field_padding", type=int, default=config.get("field_padding"))
    parser.add_argument("--poke_amplitude", type=float, default=config.get("poke_amplitude"))
    parser.add_argument("--random_seed", type=int, default=config.get("random_seed"))
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false")
    parser.set_defaults(use_gpu=config.get("use_gpu", False))
    parser.add_argument("--data_path", type=str, default=config.get("data_path"))
    
    # Extra key=value arguments
    parser.add_argument("--args", nargs="*", help="Additional key=value arguments for the function")

    args, unknown = parser.parse_known_args()

    if args.list_functions:
        print("Available functions:")
        for name in functions.keys():
            print("-", name)
        sys.exit()

    # Update params from CLI
    for key, value in vars(args).items():
        if value is not None and key != "function" and key != "args":
            params[key] = value

    # Parse extra key=value args
    kwargs = {}
    if args.args:
        for item in args.args:
            key, value = item.split("=")
            try:
                value = eval(value)
            except:
                pass
            kwargs[key] = value

    # 5 Call the chosen function
    result = functions[args.function](**kwargs)
    print("Result:", result)