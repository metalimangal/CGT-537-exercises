"""
Texture Synthesis & Inpainting Library
========================================
Implements algorithms from Szeliski Ch 10.5:
  1. Efros & Leung (1999) - pixel-by-pixel texture synthesis
  2. Wei & Levoy (2000)  - multi-resolution coarse-to-fine synthesis
  3. Criminisi et al. (2004) - priority-based exemplar inpainting
  4. Efros & Freeman (2001) - image quilting with min-error boundary cut
  5. Telea (2004) - fast marching isophote inpainting
"""

import numpy as np
from PIL import Image
import time
import sys
import json


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def load_image(path):
    """Load image as float64 numpy array [0,1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float64) / 255.0


def save_image(arr, path):
    """Save float64 [0,1] array as image."""
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img.save(path)
    return path


def gaussian_pyramid(img, levels):
    """Build a Gaussian pyramid by successive 2x downsampling."""
    from PIL import Image as PILImage
    pyramid = [img]
    for _ in range(levels - 1):
        h, w = pyramid[-1].shape[:2]
        pil = PILImage.fromarray((np.clip(pyramid[-1], 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((max(1, w // 2), max(1, h // 2)), PILImage.LANCZOS)
        pyramid.append(np.array(pil, dtype=np.float64) / 255.0)
    return pyramid


def ssd_patch(p1, p2, mask=None):
    """Sum of squared differences between two patches, optionally masked."""
    diff = p1 - p2
    if mask is not None:
        diff = diff * mask[..., np.newaxis] if diff.ndim == 3 else diff * mask
    return np.sum(diff ** 2)


def extract_patch(img, r, c, half, fill_value=0):
    """Extract a patch centered at (r,c) with radius=half, padding if needed."""
    h, w = img.shape[:2]
    size = 2 * half + 1
    if img.ndim == 3:
        patch = np.full((size, size, img.shape[2]), fill_value, dtype=img.dtype)
    else:
        patch = np.full((size, size), fill_value, dtype=img.dtype)

    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    pr0 = max(0, -r0)
    pc0 = max(0, -c0)
    pr1 = size - max(0, r1 - h)
    pc1 = size - max(0, c1 - w)

    sr0 = max(0, r0)
    sc0 = max(0, c0)
    sr1 = min(h, r1)
    sc1 = min(w, c1)

    patch[pr0:pr1, pc0:pc1] = img[sr0:sr1, sc0:sc1]
    return patch


def extract_mask_patch(mask, r, c, half):
    """Extract mask patch (1=known, 0=unknown)."""
    h, w = mask.shape
    size = 2 * half + 1
    patch = np.zeros((size, size), dtype=mask.dtype)

    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    pr0 = max(0, -r0)
    pc0 = max(0, -c0)
    pr1 = size - max(0, r1 - h)
    pc1 = size - max(0, c1 - w)

    patch[pr0:pr1, pc0:pc1] = mask[max(0, r0):min(h, r1), max(0, c0):min(w, c1)]
    return patch


# ─────────────────────────────────────────────
# 1. Efros & Leung (1999) - Pixel-based synthesis
# ─────────────────────────────────────────────

def efros_leung_synthesis(source, output_size, neighborhood=5, progress_cb=None):
    """
    Pixel-by-pixel texture synthesis.
    Grows texture in raster order by finding best matching neighborhood.
    """
    half = neighborhood // 2
    sh, sw = source.shape[:2]
    oh, ow = output_size

    # Initialize output with random seed from source
    output = np.zeros((oh, ow, 3), dtype=np.float64)
    filled = np.zeros((oh, ow), dtype=bool)

    # Seed: copy a small block from center of source
    seed_size = min(3, sh, sw, oh, ow)
    sr, sc = sh // 2, sw // 2
    orr, oc = oh // 2, ow // 2
    s2 = seed_size // 2
    output[orr-s2:orr-s2+seed_size, oc-s2:oc-s2+seed_size] = \
        source[sr-s2:sr-s2+seed_size, sc-s2:sc-s2+seed_size]
    filled[orr-s2:orr-s2+seed_size, oc-s2:oc-s2+seed_size] = True

    # Build list of unfilled pixels in raster order
    total = oh * ow
    done = int(np.sum(filled))

    # Process in raster order, multiple passes
    max_passes = 5
    for pass_num in range(max_passes):
        changed = False
        for r in range(oh):
            for c in range(ow):
                if filled[r, c]:
                    continue

                # Check if any neighbor is filled
                m = extract_mask_patch(filled.astype(np.float64), r, c, half)
                if np.sum(m) == 0:
                    continue

                # Extract neighborhood from output
                out_patch = extract_patch(output, r, c, half)

                # Find best match in source
                best_err = np.inf
                best_val = None
                candidates = []

                for sr in range(half, sh - half):
                    for sc in range(half, sw - half):
                        src_patch = source[sr-half:sr+half+1, sc-half:sc+half+1]
                        err = ssd_patch(out_patch, src_patch, m)
                        if err < best_err:
                            best_err = err
                            candidates = [(sr, sc)]
                        elif err < best_err * 1.1:
                            candidates.append((sr, sc))

                # Pick randomly from near-best candidates
                if candidates:
                    idx = np.random.randint(len(candidates))
                    br, bc = candidates[idx]
                    output[r, c] = source[br, bc]
                    filled[r, c] = True
                    changed = True
                    done += 1

                    if progress_cb and done % 100 == 0:
                        progress_cb(done / total)

        if not changed:
            break

    # Fill remaining with random source pixels
    for r in range(oh):
        for c in range(ow):
            if not filled[r, c]:
                output[r, c] = source[np.random.randint(sh), np.random.randint(sw)]

    return output


# ─────────────────────────────────────────────
# 2. Wei & Levoy (2000) - Multi-resolution synthesis
# ─────────────────────────────────────────────

def wei_levoy_synthesis(source, output_size, neighborhood=5, levels=3, progress_cb=None):
    """
    Coarse-to-fine texture synthesis using Gaussian pyramids.
    """
    half = neighborhood // 2
    src_pyramid = gaussian_pyramid(source, levels)

    # Start from coarsest level
    scale = 2 ** (levels - 1)
    coarse_h = max(2, output_size[0] // scale)
    coarse_w = max(2, output_size[1] // scale)

    # Synthesize coarsest level with simple random initialization
    coarse_src = src_pyramid[-1]
    csh, csw = coarse_src.shape[:2]
    result = np.zeros((coarse_h, coarse_w, 3))
    for r in range(coarse_h):
        for c in range(coarse_w):
            result[r, c] = coarse_src[r % csh, c % csw]

    # Refine from coarse to fine
    for level in range(levels - 2, -1, -1):
        src_level = src_pyramid[level]
        slh, slw = src_level.shape[:2]

        # Upsample result
        scale_factor = 2
        new_h = result.shape[0] * scale_factor
        new_w = result.shape[1] * scale_factor
        if level == 0:
            new_h, new_w = output_size

        upsampled = np.array(
            Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8))
            .resize((new_w, new_h), Image.LANCZOS),
            dtype=np.float64
        ) / 255.0

        result = upsampled.copy()

        # Refine each pixel
        total_pixels = new_h * new_w
        for r in range(new_h):
            for c in range(new_w):
                out_patch = extract_patch(result, r, c, half)
                m = np.ones((2 * half + 1, 2 * half + 1))

                best_err = np.inf
                best_val = None

                # Sample random candidates for efficiency
                n_samples = min(200, (slh - 2*half) * (slw - 2*half))
                for _ in range(n_samples):
                    sr = np.random.randint(half, max(half+1, slh - half))
                    sc = np.random.randint(half, max(half+1, slw - half))
                    src_patch = extract_patch(src_level, sr, sc, half)
                    err = ssd_patch(out_patch, src_patch, m)
                    if err < best_err:
                        best_err = err
                        best_val = src_level[sr, sc].copy()

                if best_val is not None:
                    result[r, c] = best_val

            if progress_cb and r % 5 == 0:
                prog = ((levels - 1 - level) + r / new_h) / levels
                progress_cb(prog)

    return result


# ─────────────────────────────────────────────
# 3. Criminisi et al. (2004) - Exemplar-based inpainting
# ─────────────────────────────────────────────

def criminisi_inpainting(image, mask, patch_size=9, progress_cb=None):
    """
    Priority-based exemplar inpainting.
    mask: 1 = hole (to fill), 0 = known
    """
    half = patch_size // 2
    result = image.copy()
    fill_mask = (1 - mask).astype(np.float64)  # 1=known, 0=hole
    h, w = image.shape[:2]

    confidence = fill_mask.copy()
    total_to_fill = int(np.sum(mask))
    filled_count = 0

    def get_boundary():
        """Find boundary pixels of the fill region."""
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(fill_mask > 0.5)
        boundary = dilated & (fill_mask < 0.5)
        return list(zip(*np.where(boundary)))

    def compute_normal(r, c):
        """Compute unit normal to the fill boundary at (r,c)."""
        patch = extract_mask_patch(fill_mask, r, c, 1)
        gy, gx = np.gradient(patch)
        nx, ny = np.mean(gx), np.mean(gy)
        mag = np.sqrt(nx**2 + ny**2) + 1e-8
        return nx / mag, ny / mag

    def compute_data_term(r, c):
        """Compute data term (strength of isophote hitting boundary)."""
        if r < 1 or r >= h-1 or c < 1 or c >= w-1:
            return 0.001
        gray = np.mean(result, axis=2) if result.ndim == 3 else result
        gy = gray[min(r+1,h-1), c] - gray[max(r-1,0), c]
        gx = gray[r, min(c+1,w-1)] - gray[r, max(c-1,0)]
        # Isophote direction (perpendicular to gradient)
        iso_x, iso_y = -gy, gx
        nx, ny = compute_normal(r, c)
        return abs(iso_x * nx + iso_y * ny) + 0.001

    iteration = 0
    max_iterations = total_to_fill + 100

    while np.any(fill_mask < 0.5) and iteration < max_iterations:
        iteration += 1
        boundary = get_boundary()
        if not boundary:
            break

        # Compute priorities
        best_priority = -1
        best_pixel = None
        for (r, c) in boundary:
            # Confidence term
            m = extract_mask_patch(confidence, r, c, half)
            conf = np.mean(m)

            # Data term
            data = compute_data_term(r, c)

            priority = conf * data
            if priority > best_priority:
                best_priority = priority
                best_pixel = (r, c)

        if best_pixel is None:
            break

        r, c = best_pixel

        # Extract patch and mask
        out_patch = extract_patch(result, r, c, half)
        m = extract_mask_patch(fill_mask, r, c, half)

        # Find best matching patch in known region
        best_err = np.inf
        best_r, best_c = 0, 0

        # Search over source
        step = max(1, min(h, w) // 50)
        for sr in range(half, h - half, step):
            for sc in range(half, w - half, step):
                # Only consider fully known patches
                sm = extract_mask_patch(fill_mask, sr, sc, half)
                if np.mean(sm) < 0.99:
                    continue
                src_patch = extract_patch(result, sr, sc, half)
                err = ssd_patch(out_patch, src_patch, m)
                if err < best_err:
                    best_err = err
                    best_r, best_c = sr, sc

        # Refine search around best match
        for sr in range(max(half, best_r - 5), min(h - half, best_r + 6)):
            for sc in range(max(half, best_c - 5), min(w - half, best_c + 6)):
                sm = extract_mask_patch(fill_mask, sr, sc, half)
                if np.mean(sm) < 0.99:
                    continue
                src_patch = extract_patch(result, sr, sc, half)
                err = ssd_patch(out_patch, src_patch, m)
                if err < best_err:
                    best_err = err
                    best_r, best_c = sr, sc

        # Copy unfilled pixels from best match
        src_patch = extract_patch(result, best_r, best_c, half)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and fill_mask[rr, cc] < 0.5:
                    result[rr, cc] = src_patch[dr + half, dc + half]
                    fill_mask[rr, cc] = 1.0
                    confidence[rr, cc] = confidence[r, c]
                    filled_count += 1

        if progress_cb:
            progress_cb(min(1.0, filled_count / max(1, total_to_fill)))

    return result


# ─────────────────────────────────────────────
# 4. Efros & Freeman (2001) - Image Quilting
# ─────────────────────────────────────────────

def min_error_boundary_cut(overlap_err):
    """
    Find minimum error boundary cut through an overlap region
    using dynamic programming.
    overlap_err: 2D array of squared differences in overlap zone.
    Returns: binary mask (1 = use new patch, 0 = keep old).
    """
    h, w = overlap_err.shape
    if h == 0 or w == 0:
        return np.ones_like(overlap_err)

    # Dynamic programming - accumulate min cost
    dp = overlap_err.copy()
    for i in range(1, h):
        for j in range(w):
            candidates = [dp[i-1, j]]
            if j > 0:
                candidates.append(dp[i-1, j-1])
            if j < w - 1:
                candidates.append(dp[i-1, j+1])
            dp[i, j] += min(candidates)

    # Trace back
    mask = np.zeros((h, w), dtype=np.float64)
    j = np.argmin(dp[-1])
    for i in range(h-1, -1, -1):
        mask[i, j:] = 1
        if i > 0:
            candidates = [(dp[i-1, j], j)]
            if j > 0:
                candidates.append((dp[i-1, j-1], j-1))
            if j < w - 1:
                candidates.append((dp[i-1, j+1], j+1))
            _, j = min(candidates, key=lambda x: x[0])

    return mask


def image_quilting(source, output_size, block_size=32, overlap=8, progress_cb=None):
    """
    Image quilting with minimum error boundary cut.
    Tiles patches from source with optimized seams.
    """
    sh, sw = source.shape[:2]
    oh, ow = output_size
    output = np.zeros((oh, ow, 3), dtype=np.float64)

    bs = min(block_size, sh, sw)
    ov = min(overlap, bs // 4)
    step = bs - ov

    n_rows = (oh - ov) // step + 1
    n_cols = (ow - ov) // step + 1
    total = n_rows * n_cols
    count = 0

    for bi in range(n_rows):
        for bj in range(n_cols):
            y = bi * step
            x = bj * step

            if bi == 0 and bj == 0:
                # First block: random
                sr = np.random.randint(0, max(1, sh - bs))
                sc = np.random.randint(0, max(1, sw - bs))
                patch = source[sr:sr+bs, sc:sc+bs].copy()
                ph, pw = patch.shape[:2]
                output[y:y+ph, x:x+pw] = patch
                count += 1
                continue

            # Find best matching patch considering overlap
            best_err = np.inf
            best_patch = None
            n_cand = min(100, (sh - bs) * (sw - bs))

            for _ in range(n_cand):
                sr = np.random.randint(0, max(1, sh - bs))
                sc = np.random.randint(0, max(1, sw - bs))
                candidate = source[sr:sr+bs, sc:sc+bs]
                ch, cw = candidate.shape[:2]
                if ch < bs or cw < bs:
                    continue

                err = 0
                # Left overlap
                if bj > 0 and x + ov <= ow:
                    oy_end = min(bs, oh - y)
                    left_existing = output[y:y+oy_end, x:x+ov]
                    left_cand = candidate[:oy_end, :ov]
                    err += np.sum((left_existing - left_cand) ** 2)

                # Top overlap
                if bi > 0 and y + ov <= oh:
                    ox_end = min(bs, ow - x)
                    top_existing = output[y:y+ov, x:x+ox_end]
                    top_cand = candidate[:ov, :ox_end]
                    err += np.sum((top_existing - top_cand) ** 2)

                if err < best_err:
                    best_err = err
                    best_patch = candidate.copy()

            if best_patch is None:
                sr = np.random.randint(0, max(1, sh - bs))
                sc = np.random.randint(0, max(1, sw - bs))
                best_patch = source[sr:sr+bs, sc:sc+bs].copy()

            # Apply with minimum error boundary cut
            ph, pw = min(bs, oh - y), min(bs, ow - x)
            patch_region = best_patch[:ph, :pw]
            existing = output[y:y+ph, x:x+pw]

            mask = np.ones((ph, pw), dtype=np.float64)

            # Left overlap seam
            if bj > 0 and ov > 0:
                ov_w = min(ov, pw)
                left_err = np.sum((existing[:, :ov_w] - patch_region[:, :ov_w]) ** 2, axis=2)
                left_mask = min_error_boundary_cut(left_err)
                mask[:, :ov_w] = left_mask

            # Top overlap seam
            if bi > 0 and ov > 0:
                ov_h = min(ov, ph)
                top_err = np.sum((existing[:ov_h, :] - patch_region[:ov_h, :]) ** 2, axis=2)
                top_mask = min_error_boundary_cut(top_err.T).T
                mask[:ov_h, :] = np.minimum(mask[:ov_h, :], top_mask)

            # Blend using mask
            mask_3d = mask[..., np.newaxis]
            output[y:y+ph, x:x+pw] = existing * (1 - mask_3d) + patch_region * mask_3d

            count += 1
            if progress_cb:
                progress_cb(count / total)

    return output[:oh, :ow]


# ─────────────────────────────────────────────
# 5. Telea (2004) - Fast Marching Inpainting
# ─────────────────────────────────────────────

def telea_inpainting(image, mask, radius=5, progress_cb=None):
    """
    Fast marching method inpainting (Telea 2004).
    Fills holes by propagating isophotes (level lines) inward.
    mask: 1 = hole, 0 = known
    """
    result = image.copy()
    h, w = image.shape[:2]
    known = (mask < 0.5).astype(np.float64)  # 1=known
    to_fill = mask.copy().astype(np.float64)

    # Distance transform for ordering
    from scipy.ndimage import distance_transform_edt, binary_dilation
    dist = distance_transform_edt(mask > 0.5)

    # Process in order of distance (outside-in)
    fill_coords = list(zip(*np.where(mask > 0.5)))
    fill_coords.sort(key=lambda p: dist[p[0], p[1]])

    total = len(fill_coords)
    for idx, (r, c) in enumerate(fill_coords):
        # Weighted average of known neighbors within radius
        r_start = max(0, r - radius)
        r_end = min(h, r + radius + 1)
        c_start = max(0, c - radius)
        c_end = min(w, c + radius + 1)

        total_w = 0
        color_sum = np.zeros(3)

        for nr in range(r_start, r_end):
            for nc in range(c_start, c_end):
                if known[nr, nc] < 0.5:
                    continue
                dr, dc = nr - r, nc - c
                d = np.sqrt(dr**2 + dc**2)
                if d > radius or d == 0:
                    continue

                # Direction weight (favor isophote direction)
                w_d = 1.0 / (d * d + 1e-6)

                # Level set weight
                w_l = 1.0 / (1.0 + abs(dist[nr, nc] - dist[r, c]))

                weight = w_d * w_l
                color_sum += weight * result[nr, nc]
                total_w += weight

        if total_w > 0:
            result[r, c] = color_sum / total_w
        else:
            # Fallback: nearest known pixel
            for rad in range(1, max(h, w)):
                found = False
                for dr in range(-rad, rad + 1):
                    for dc in range(-rad, rad + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and known[nr, nc] > 0.5:
                            result[r, c] = result[nr, nc]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        known[r, c] = 1.0

        if progress_cb and idx % 50 == 0:
            progress_cb(idx / total)

    return result


# ─────────────────────────────────────────────
# CLI entry point for batch processing
# ─────────────────────────────────────────────

def create_sample_texture(size=64):
    """Create a sample procedural texture for testing."""
    h, w = size, size
    img = np.zeros((h, w, 3))
    for r in range(h):
        for c in range(w):
            # Checkerboard + noise
            check = ((r // 8) + (c // 8)) % 2
            noise = np.random.rand(3) * 0.1
            if check:
                img[r, c] = np.array([0.8, 0.6, 0.3]) + noise
            else:
                img[r, c] = np.array([0.3, 0.5, 0.7]) + noise
    return np.clip(img, 0, 1)


def create_brick_texture(size=64):
    """Create a brick-like texture."""
    h, w = size, size
    img = np.zeros((h, w, 3))
    brick_h, brick_w = 8, 16
    mortar = np.array([0.7, 0.7, 0.65])

    for r in range(h):
        for c in range(w):
            row = r // brick_h
            offset = (brick_w // 2) * (row % 2)
            in_mortar = (r % brick_h < 1) or ((c + offset) % brick_w < 1)
            noise = np.random.rand(3) * 0.05
            if in_mortar:
                img[r, c] = mortar + noise
            else:
                brick_color = np.array([0.7, 0.3, 0.2]) + noise
                # Vary slightly per brick
                brick_id = row * 100 + ((c + offset) // brick_w)
                np.random.seed(brick_id)
                variation = np.random.rand(3) * 0.1 - 0.05
                np.random.seed(None)
                img[r, c] = brick_color + variation

    return np.clip(img, 0, 1)


def create_circle_mask(h, w, center=None, radius=None):
    """Create a circular mask (1=hole)."""
    if center is None:
        center = (h // 2, w // 2)
    if radius is None:
        radius = min(h, w) // 6
    mask = np.zeros((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            if (r - center[0])**2 + (c - center[1])**2 < radius**2:
                mask[r, c] = 1.0
    return mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Texture Synthesis & Inpainting")
    parser.add_argument("--mode", choices=["synthesis", "quilting", "inpaint_criminisi",
                                            "inpaint_telea", "multirez", "demo"],
                        default="demo")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--size", type=int, default=96, help="Output size (square)")
    parser.add_argument("--patch", type=int, default=9, help="Patch/neighborhood size")
    parser.add_argument("--block", type=int, default=24, help="Block size for quilting")
    parser.add_argument("--overlap", type=int, default=6, help="Overlap for quilting")
    parser.add_argument("--levels", type=int, default=3, help="Pyramid levels")

    args = parser.parse_args()

    def print_progress(p):
        bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
        print(f"\r  [{bar}] {p*100:.1f}%", end="", flush=True)

    if args.source:
        source = load_image(args.source)

        if args.mode == "synthesis":
            print("Running Efros & Leung synthesis...")
            result = efros_leung_synthesis(source, (args.size, args.size),
                                          neighborhood=args.patch, progress_cb=print_progress)
        elif args.mode == "quilting":
            print("Running image quilting...")
            result = image_quilting(source, (args.size, args.size),
                                   block_size=args.block, overlap=args.overlap,
                                   progress_cb=print_progress)
        elif args.mode == "multirez":
            print("Running multi-resolution synthesis...")
            result = wei_levoy_synthesis(source, (args.size, args.size),
                                        neighborhood=args.patch, levels=args.levels,
                                        progress_cb=print_progress)
        elif args.mode in ("inpaint_criminisi", "inpaint_telea"):
            if args.mask:
                mask_img = load_image(args.mask)
                mask = np.mean(mask_img, axis=2) > 0.5
            else:
                h, w = source.shape[:2]
                mask = create_circle_mask(h, w)

            if args.mode == "inpaint_criminisi":
                print("Running Criminisi inpainting...")
                result = criminisi_inpainting(source, mask.astype(np.float64),
                                             patch_size=args.patch, progress_cb=print_progress)
            else:
                print("Running Telea inpainting...")
                result = telea_inpainting(source, mask.astype(np.float64),
                                         radius=5, progress_cb=print_progress)
        elif args.mode == "demo":
            print("=== Texture Synthesis & Inpainting Demo ===\n")

            # Create source textures
            checker = create_sample_texture(64)
            save_image(checker, "sample_checker.png")
            brick = create_brick_texture(64)
            save_image(brick, "sample_brick.png")

            source = load_image(args.source)

            # --- Individual Demos ---
            print("=== Individual Demos ===\n")

            # Efros & Leung synthesis
            print("1. Efros & Leung synthesis...")
            result = efros_leung_synthesis(source, (48, 48), neighborhood=5, progress_cb=print_progress)
            save_image(result, "indiv_efros_leung.png")
            print("\n   Saved: indiv_efros_leung.png")

            # Image quilting
            print("2. Efros & Freeman image quilting...")
            result = image_quilting(source, (128, 128), block_size=24, overlap=6, progress_cb=print_progress)
            save_image(result, "indiv_quilting.png")
            print("\n   Saved: indiv_quilting.png")

            # Criminisi inpainting
            print("3. Criminisi inpainting...")
            test_img = source.copy()
            test_img = np.array(Image.fromarray((test_img * 255).astype(np.uint8))
                        .resize((96, 96), Image.LANCZOS), dtype=np.float64)/255.0
            mask = create_circle_mask(96, 96, radius=12)
            test_img[mask > 0.5] = 0
            result = criminisi_inpainting(test_img, mask, patch_size=9, progress_cb=print_progress)
            save_image(result, "indiv_criminisi.png")
            print("\n   Saved: indiv_criminisi.png")

            # Telea inpainting
            print("4. Telea inpainting...")
            test_img2 = source.copy()
            test_img2 = np.array(Image.fromarray((test_img2 * 255).astype(np.uint8))
                         .resize((96, 96), Image.LANCZOS), dtype=np.float64)/255.0
            test_img2[mask > 0.5] = 0
            result = telea_inpainting(test_img2, mask, radius=5, progress_cb=print_progress)
            save_image(result, "indiv_telea.png")
            print("\n   Saved: indiv_telea.png")

            # Multi-resolution synthesis
            print("5. Wei & Levoy multi-resolution synthesis...")
            result = wei_levoy_synthesis(source, (64, 64), neighborhood=5, levels=2, progress_cb=print_progress)
            save_image(result, "indiv_multirez.png")
            print("\n   Saved: indiv_multirez.png")

            # --- Pipeline Demo ---
            print("\n=== Pipeline Demo (cumulative) ===\n")
            pipeline_img = source.copy()

            # Step 1: Efros & Leung
            print("Step 1: Efros & Leung synthesis...")
            pipeline_img = efros_leung_synthesis(pipeline_img, (64, 64), neighborhood=5, progress_cb=print_progress)
            save_image(pipeline_img, "pipeline_1_efros_leung.png")

            # Step 2: Multi-resolution
            print("Step 2: Multi-resolution synthesis...")
            pipeline_img = wei_levoy_synthesis(pipeline_img, (64, 64), neighborhood=5, levels=2, progress_cb=print_progress)
            save_image(pipeline_img, "pipeline_2_multirez.png")

            # Step 3: Criminisi inpainting
            print("Step 3: Criminisi inpainting...")
            mask = create_circle_mask(64, 64, radius=12)
            pipeline_img[mask > 0.5] = 0
            pipeline_img = criminisi_inpainting(pipeline_img, mask.astype(np.float64), patch_size=5, progress_cb=print_progress)
            save_image(pipeline_img, "pipeline_3_criminisi.png")

            # Step 4: Telea inpainting
            print("Step 4: Telea inpainting...")
            pipeline_img[mask > 0.5] = 0
            pipeline_img = telea_inpainting(pipeline_img, mask.astype(np.float64), radius=5, progress_cb=print_progress)
            save_image(pipeline_img, "pipeline_4_telea.png")

            # Step 5: Image quilting
            print("Step 5: Image quilting...")
            pipeline_img = image_quilting(pipeline_img, (128, 128), block_size=24, overlap=6, progress_cb=print_progress)
            save_image(pipeline_img, "pipeline_5_quilting.png")

            print("\n=== Demo Complete! ===")

    


        save_image(result, args.output)
        print(f"\nSaved: {args.output}")
    else:
        print("Please specify --source or use --mode demo")
