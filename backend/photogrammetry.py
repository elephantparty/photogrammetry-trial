"""
Photogrammetry Pipeline v2
Improved 3D reconstruction with better feature matching and filtering.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Callable, Optional, Dict, Tuple
import json
from dataclasses import dataclass
import struct


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class ImageData:
    """Processed image data."""
    idx: int
    image: np.ndarray
    gray: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    shape: Tuple[int, int]


class PhotogrammetryPipeline:
    """
    Improved photogrammetry pipeline for 3D reconstruction.
    """

    def __init__(
        self,
        image_paths: List[str],
        output_dir: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.progress_callback = progress_callback or (lambda p, m: None)

        # Results
        self.images: List[ImageData] = []
        self.matches: Dict[Tuple[int, int], np.ndarray] = {}
        self.points_3d: np.ndarray = None
        self.point_colors: np.ndarray = None
        self.intrinsics: CameraIntrinsics = None
        self.triangles = None

        # Use SIFT with more features for better matching
        self.detector = cv2.SIFT_create(
            nfeatures=8000,
            contrastThreshold=0.02,
            edgeThreshold=20
        )

        # FLANN matcher for faster, better matching
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=100)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def run(self) -> Dict:
        """Execute the photogrammetry pipeline."""
        try:
            self.progress_callback(5, "Loading images...")
            self._load_images()

            self.progress_callback(15, "Detecting features...")
            self._detect_features()

            self.progress_callback(30, "Matching features across images...")
            self._match_features()

            self.progress_callback(50, "Reconstructing 3D structure...")
            self._reconstruct()

            self.progress_callback(75, "Filtering and refining point cloud...")
            self._filter_points()

            self.progress_callback(85, "Generating mesh...")
            self._generate_mesh()

            self.progress_callback(92, "Exporting model...")
            self._export_glb()
            self._generate_preview()
            self._export_ply()

            self.progress_callback(100, "Complete!")

            return {
                "success": True,
                "points": len(self.points_3d) if self.points_3d is not None else 0
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def _load_images(self):
        """Load and preprocess images to consistent size."""
        target_size = None

        for idx, path in enumerate(self.image_paths):
            img = cv2.imread(str(path))
            if img is None:
                continue

            # Resize large images
            h, w = img.shape[:2]
            max_dim = 1600
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Store first image size as target
            if target_size is None:
                target_size = img.shape[:2]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.images.append(ImageData(
                idx=idx,
                image=img,
                gray=gray,
                keypoints=[],
                descriptors=None,
                shape=img.shape[:2]
            ))

        if len(self.images) < 3:
            raise ValueError(f"Need at least 3 valid images, got {len(self.images)}")

        # Estimate camera intrinsics from image size
        h, w = self.images[0].shape
        focal = max(h, w) * 1.2
        self.intrinsics = CameraIntrinsics(fx=focal, fy=focal, cx=w/2, cy=h/2)

    def _detect_features(self):
        """Detect SIFT features with adaptive thresholding."""
        for img_data in self.images:
            # Apply CLAHE for better feature detection
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(img_data.gray)

            kp, desc = self.detector.detectAndCompute(enhanced, None)

            if desc is not None and len(kp) >= 100:
                img_data.keypoints = kp
                img_data.descriptors = desc
            else:
                # Retry with lower threshold
                detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)
                kp, desc = detector.detectAndCompute(enhanced, None)
                img_data.keypoints = kp if kp else []
                img_data.descriptors = desc

        # Filter images without enough features
        self.images = [img for img in self.images if img.descriptors is not None and len(img.keypoints) >= 50]

        if len(self.images) < 3:
            raise ValueError("Not enough features detected in images")

    def _match_features(self):
        """Match features between all image pairs with geometric verification."""
        n = len(self.images)
        K = self.intrinsics.matrix()

        for i in range(n):
            for j in range(i + 1, n):
                img_i = self.images[i]
                img_j = self.images[j]

                if img_i.descriptors is None or img_j.descriptors is None:
                    continue

                # KNN matching
                try:
                    raw_matches = self.matcher.knnMatch(img_i.descriptors, img_j.descriptors, k=2)
                except cv2.error:
                    continue

                # Ratio test
                good = []
                for m_list in raw_matches:
                    if len(m_list) == 2:
                        m, n_match = m_list
                        if m.distance < 0.7 * n_match.distance:
                            good.append(m)

                if len(good) < 20:
                    continue

                # Get point coordinates
                pts_i = np.float32([img_i.keypoints[m.queryIdx].pt for m in good])
                pts_j = np.float32([img_j.keypoints[m.trainIdx].pt for m in good])

                # Geometric verification with fundamental matrix
                F, mask = cv2.findFundamentalMat(pts_i, pts_j, cv2.FM_RANSAC, 2.0, 0.99)

                if F is None or mask is None:
                    continue

                mask = mask.ravel().astype(bool)
                inlier_count = np.sum(mask)

                if inlier_count >= 15:
                    # Store verified matches
                    verified_matches = np.array([
                        [good[k].queryIdx, good[k].trainIdx]
                        for k in range(len(good)) if mask[k]
                    ])
                    self.matches[(i, j)] = verified_matches

        if len(self.matches) < 2:
            raise ValueError("Not enough matching image pairs found")

    def _reconstruct(self):
        """Incremental structure from motion reconstruction."""
        K = self.intrinsics.matrix()

        # Find best initial pair (most matches)
        best_pair = max(self.matches.keys(), key=lambda k: len(self.matches[k]))
        i, j = best_pair

        # Initialize with first pair
        matches = self.matches[best_pair]
        pts_i = np.float32([self.images[i].keypoints[m[0]].pt for m in matches])
        pts_j = np.float32([self.images[j].keypoints[m[1]].pt for m in matches])

        # Find essential matrix and recover pose
        E, mask = cv2.findEssentialMat(pts_i, pts_j, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is None:
            raise ValueError("Could not compute essential matrix")

        mask = mask.ravel().astype(bool)
        _, R, t, pose_mask = cv2.recoverPose(E, pts_i[mask], pts_j[mask], K)

        # Projection matrices
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])

        # Triangulate initial points
        pts_i_masked = pts_i[mask]
        pts_j_masked = pts_j[mask]

        points_4d = cv2.triangulatePoints(P1, P2, pts_i_masked.T, pts_j_masked.T)
        points_3d = (points_4d[:3] / points_4d[3]).T

        # Filter by reprojection error and depth
        valid_mask = self._filter_triangulated_points(points_3d, pts_i_masked, pts_j_masked, P1, P2)
        points_3d = points_3d[valid_mask]

        # Get colors from first image
        colors = []
        for pt2d in pts_i_masked[valid_mask]:
            x, y = int(pt2d[0]), int(pt2d[1])
            img = self.images[i].image
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                colors.append(img[y, x][::-1] / 255.0)
            else:
                colors.append([0.5, 0.5, 0.5])

        all_points = list(points_3d)
        all_colors = list(colors)

        # Add points from other image pairs
        camera_poses = {i: np.eye(4), j: np.eye(4)}
        camera_poses[j][:3, :3] = R
        camera_poses[j][:3, 3] = t.flatten()

        for (idx_a, idx_b), matches in self.matches.items():
            if (idx_a, idx_b) == best_pair:
                continue

            # Get points
            pts_a = np.float32([self.images[idx_a].keypoints[m[0]].pt for m in matches])
            pts_b = np.float32([self.images[idx_b].keypoints[m[1]].pt for m in matches])

            # Try to find essential matrix
            E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is None:
                continue

            mask = mask.ravel().astype(bool)
            if np.sum(mask) < 10:
                continue

            _, R_ab, t_ab, _ = cv2.recoverPose(E, pts_a[mask], pts_b[mask], K)

            # Triangulate
            P_a = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P_b = K @ np.hstack([R_ab, t_ab])

            pts_a_masked = pts_a[mask]
            pts_b_masked = pts_b[mask]

            points_4d = cv2.triangulatePoints(P_a, P_b, pts_a_masked.T, pts_b_masked.T)
            new_points = (points_4d[:3] / points_4d[3]).T

            # Filter
            valid = self._filter_triangulated_points(new_points, pts_a_masked, pts_b_masked, P_a, P_b)
            new_points = new_points[valid]

            # Colors
            for pt2d in pts_a_masked[valid]:
                x, y = int(pt2d[0]), int(pt2d[1])
                img = self.images[idx_a].image
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    all_colors.append(img[y, x][::-1] / 255.0)
                else:
                    all_colors.append([0.5, 0.5, 0.5])

            all_points.extend(new_points)

        if len(all_points) < 10:
            raise ValueError(f"Only {len(all_points)} points reconstructed, need more overlap between photos")

        self.points_3d = np.array(all_points)
        self.point_colors = np.array(all_colors)

    def _filter_triangulated_points(self, points_3d, pts1, pts2, P1, P2, max_reproj_error=5.0):
        """Filter points by reprojection error and depth."""
        valid = np.ones(len(points_3d), dtype=bool)

        for idx, (pt3d, pt1, pt2) in enumerate(zip(points_3d, pts1, pts2)):
            # Check if point is in front of both cameras
            pt_h = np.append(pt3d, 1)

            # Depth check (z > 0 in camera coordinates)
            z1 = pt3d[2]
            z2 = (P2 @ pt_h)[2]

            if z1 <= 0.1 or z2 <= 0.1:
                valid[idx] = False
                continue

            # Reprojection error
            proj1 = P1 @ pt_h
            proj1 = proj1[:2] / proj1[2]
            err1 = np.linalg.norm(proj1 - pt1)

            proj2 = P2 @ pt_h
            proj2 = proj2[:2] / proj2[2]
            err2 = np.linalg.norm(proj2 - pt2)

            if err1 > max_reproj_error or err2 > max_reproj_error:
                valid[idx] = False
                continue

            # Check for extreme distances
            if np.linalg.norm(pt3d) > 100:
                valid[idx] = False

        return valid

    def _filter_points(self):
        """Statistical outlier removal and normalization."""
        if len(self.points_3d) < 10:
            return

        points = self.points_3d
        colors = self.point_colors

        # Remove statistical outliers (points far from neighbors)
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        k = min(20, len(points) - 1)
        distances, _ = tree.query(points, k=k+1)
        mean_distances = distances[:, 1:].mean(axis=1)

        threshold = np.mean(mean_distances) + 2 * np.std(mean_distances)
        inlier_mask = mean_distances < threshold

        points = points[inlier_mask]
        colors = colors[inlier_mask]

        if len(points) < 10:
            raise ValueError("Too few points after filtering")

        # Center and normalize
        centroid = np.median(points, axis=0)
        points = points - centroid

        # Scale to unit sphere
        scale = np.percentile(np.linalg.norm(points, axis=1), 95)
        if scale > 0:
            points = points / scale

        self.points_3d = points
        self.point_colors = colors

    def _generate_mesh(self):
        """Generate mesh using Delaunay triangulation with better filtering."""
        if len(self.points_3d) < 4:
            self.triangles = None
            return

        try:
            from scipy.spatial import Delaunay

            # Project to 2D for triangulation
            points_2d = self.points_3d[:, :2]

            tri = Delaunay(points_2d)
            triangles = tri.simplices

            # Filter triangles by edge length and aspect ratio
            valid = []
            median_dist = np.median(np.linalg.norm(
                self.points_3d[triangles[:, 0]] - self.points_3d[triangles[:, 1]], axis=1
            ))
            max_edge = median_dist * 3

            for t in triangles:
                p0, p1, p2 = self.points_3d[t]
                edges = [
                    np.linalg.norm(p1 - p0),
                    np.linalg.norm(p2 - p1),
                    np.linalg.norm(p0 - p2)
                ]

                if max(edges) < max_edge and min(edges) > 0.001:
                    # Check aspect ratio
                    if max(edges) / min(edges) < 10:
                        valid.append(t)

            self.triangles = np.array(valid) if valid else None

        except Exception as e:
            print(f"Mesh generation failed: {e}")
            self.triangles = None

    def _export_glb(self):
        """Export to GLB format."""
        output_path = self.output_dir / "model.glb"

        points = self.points_3d.astype(np.float32)
        colors = (np.clip(self.point_colors, 0, 1) * 255).astype(np.uint8)

        min_pos = points.min(axis=0).tolist()
        max_pos = points.max(axis=0).tolist()

        position_data = points.tobytes()
        color_data = colors.tobytes()

        has_mesh = self.triangles is not None and len(self.triangles) > 0

        if has_mesh:
            indices = self.triangles.flatten().astype(np.uint32)
            indices_data = indices.tobytes()
            buffer_data = position_data + color_data + indices_data
        else:
            buffer_data = position_data + color_data

        gltf = {
            "asset": {"version": "2.0", "generator": "Photogrammetry Pipeline v2"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "buffers": [{"byteLength": len(buffer_data)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(position_data), "target": 34962},
                {"buffer": 0, "byteOffset": len(position_data), "byteLength": len(color_data), "target": 34962}
            ],
            "accessors": [
                {
                    "bufferView": 0, "byteOffset": 0, "componentType": 5126,
                    "count": len(points), "type": "VEC3", "min": min_pos, "max": max_pos
                },
                {
                    "bufferView": 1, "byteOffset": 0, "componentType": 5121,
                    "normalized": True, "count": len(colors), "type": "VEC3"
                }
            ],
            "materials": [{"pbrMetallicRoughness": {"metallicFactor": 0, "roughnessFactor": 1}, "doubleSided": True}]
        }

        if has_mesh:
            gltf["bufferViews"].append({
                "buffer": 0,
                "byteOffset": len(position_data) + len(color_data),
                "byteLength": len(indices_data),
                "target": 34963
            })
            gltf["accessors"].append({
                "bufferView": 2, "byteOffset": 0, "componentType": 5125,
                "count": len(indices), "type": "SCALAR"
            })
            gltf["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0, "mode": 4}]}]
        else:
            gltf["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "material": 0, "mode": 0}]}]

        # Write GLB
        json_str = json.dumps(gltf, separators=(',', ':'))
        while len(json_str) % 4 != 0:
            json_str += ' '
        json_bytes = json_str.encode('utf-8')

        while len(buffer_data) % 4 != 0:
            buffer_data += b'\x00'

        total_length = 12 + 8 + len(json_bytes) + 8 + len(buffer_data)

        with open(output_path, 'wb') as f:
            f.write(b'glTF')
            f.write(struct.pack('<I', 2))
            f.write(struct.pack('<I', total_length))
            f.write(struct.pack('<I', len(json_bytes)))
            f.write(b'JSON')
            f.write(json_bytes)
            f.write(struct.pack('<I', len(buffer_data)))
            f.write(b'BIN\x00')
            f.write(buffer_data)

    def _generate_preview(self):
        """Generate preview image."""
        preview_path = self.output_dir / "preview.png"
        size = 512
        preview = np.ones((size, size, 3), dtype=np.uint8) * 30

        if self.points_3d is None or len(self.points_3d) == 0:
            cv2.imwrite(str(preview_path), preview)
            return

        points = self.points_3d.copy()

        # Rotate for better view
        angle = np.pi / 5
        Rx = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        Ry = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        points = (Ry @ Rx @ points.T).T

        # Project
        focal = 250
        cx, cy = size // 2, size // 2

        z_order = np.argsort(-points[:, 2])

        for idx in z_order:
            pt = points[idx]
            if pt[2] < 0.1:
                continue

            x = int(focal * pt[0] / (pt[2] + 1) + cx)
            y = int(focal * pt[1] / (pt[2] + 1) + cy)

            if 0 <= x < size and 0 <= y < size:
                color = (self.point_colors[idx] * 255).astype(int).tolist()
                cv2.circle(preview, (x, y), 2, color, -1)

        cv2.imwrite(str(preview_path), preview)

    def _export_ply(self):
        """Export point cloud as PLY."""
        ply_path = self.output_dir / "model.ply"

        with open(ply_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3d)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")

            for pt, color in zip(self.points_3d, self.point_colors):
                r, g, b = (np.clip(color, 0, 1) * 255).astype(int)
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")
