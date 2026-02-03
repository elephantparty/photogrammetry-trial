"""
Photogrammetry Pipeline
Reconstructs 3D models from multiple 2D images using Structure from Motion (SfM).
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
    fx: float  # focal length x
    fy: float  # focal length y
    cx: float  # principal point x
    cy: float  # principal point y

    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)


@dataclass
class ImageFeatures:
    """Detected features for an image."""
    image_idx: int
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]


class PhotogrammetryPipeline:
    """
    Complete photogrammetry pipeline for 3D reconstruction.

    Steps:
    1. Load and preprocess images
    2. Detect features (SIFT)
    3. Match features between image pairs
    4. Estimate camera poses using Essential matrix
    5. Triangulate 3D points
    6. Bundle adjustment (simplified)
    7. Generate dense point cloud
    8. Create mesh and export to GLB
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

        # Pipeline state
        self.images: List[np.ndarray] = []
        self.image_features: List[ImageFeatures] = []
        self.matches: Dict[Tuple[int, int], List] = {}
        self.camera_poses: List[np.ndarray] = []
        self.points_3d: np.ndarray = None
        self.point_colors: np.ndarray = None
        self.intrinsics: CameraIntrinsics = None

        # Feature detector
        self.feature_detector = cv2.SIFT_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def run(self) -> Dict:
        """Execute the full photogrammetry pipeline."""
        try:
            # Step 1: Load images
            self.progress_callback(10, "Loading and analyzing images...")
            self._load_images()

            # Step 2: Detect features
            self.progress_callback(20, "Detecting image features...")
            self._detect_features()

            # Step 3: Match features
            self.progress_callback(35, "Matching features between images...")
            self._match_features()

            # Step 4: Estimate camera poses
            self.progress_callback(50, "Estimating camera positions...")
            self._estimate_poses()

            # Step 5: Triangulate points
            self.progress_callback(65, "Triangulating 3D points...")
            self._triangulate_points()

            # Step 6: Dense reconstruction
            self.progress_callback(75, "Generating dense point cloud...")
            self._dense_reconstruction()

            # Step 7: Generate mesh
            self.progress_callback(85, "Creating 3D mesh...")
            self._generate_mesh()

            # Step 8: Export to GLB
            self.progress_callback(95, "Exporting 3D model...")
            self._export_glb()

            # Generate preview
            self._generate_preview()

            self.progress_callback(100, "Complete!")

            return {
                "success": True,
                "points": len(self.points_3d) if self.points_3d is not None else 0,
                "model_path": str(self.output_dir / "model.glb")
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _load_images(self):
        """Load and preprocess all images."""
        for path in self.image_paths:
            img = cv2.imread(str(path))
            if img is None:
                continue

            # Resize if too large (max 1920px on longest side)
            max_dim = 1920
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale)

            self.images.append(img)

        if len(self.images) < 3:
            raise ValueError("Need at least 3 valid images")

        # Estimate camera intrinsics from first image
        h, w = self.images[0].shape[:2]
        focal = max(h, w) * 1.2  # Rough estimate
        self.intrinsics = CameraIntrinsics(
            fx=focal, fy=focal,
            cx=w / 2, cy=h / 2
        )

    def _detect_features(self):
        """Detect SIFT features in all images."""
        for idx, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)

            if descriptors is None or len(keypoints) < 10:
                continue

            self.image_features.append(ImageFeatures(
                image_idx=idx,
                keypoints=keypoints,
                descriptors=descriptors,
                image_shape=img.shape[:2]
            ))

    def _match_features(self):
        """Match features between image pairs."""
        n = len(self.image_features)

        for i in range(n):
            for j in range(i + 1, n):
                feat_i = self.image_features[i]
                feat_j = self.image_features[j]

                # KNN matching with ratio test
                matches = self.matcher.knnMatch(
                    feat_i.descriptors,
                    feat_j.descriptors,
                    k=2
                )

                # Apply Lowe's ratio test
                good_matches = []
                for m_list in matches:
                    if len(m_list) == 2:
                        m, n_match = m_list
                        if m.distance < 0.75 * n_match.distance:
                            good_matches.append(m)

                if len(good_matches) >= 20:
                    self.matches[(i, j)] = good_matches

    def _estimate_poses(self):
        """Estimate camera poses using essential matrix decomposition."""
        K = self.intrinsics.matrix()
        n = len(self.image_features)

        # Initialize first camera at origin
        self.camera_poses = [np.eye(4, dtype=np.float64)]

        # Process subsequent cameras
        for i in range(1, n):
            best_pose = None
            best_inliers = 0

            # Try matching with previous cameras
            for j in range(i):
                key = (j, i) if (j, i) in self.matches else (i, j)
                if key not in self.matches:
                    continue

                matches = self.matches[key]
                feat_i = self.image_features[j]
                feat_j = self.image_features[i]

                # Get matched points
                pts1 = np.float64([feat_i.keypoints[m.queryIdx].pt for m in matches])
                pts2 = np.float64([feat_j.keypoints[m.trainIdx].pt for m in matches])

                # Find essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is None:
                    continue

                inliers = np.sum(mask)
                if inliers < 20:
                    continue

                # Recover pose
                _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

                if inliers > best_inliers:
                    best_inliers = inliers
                    # Compose with reference camera pose
                    ref_pose = self.camera_poses[j]
                    best_pose = np.eye(4, dtype=np.float64)
                    best_pose[:3, :3] = R @ ref_pose[:3, :3]
                    best_pose[:3, 3] = ref_pose[:3, :3].T @ t.flatten() + ref_pose[:3, 3]

            if best_pose is not None:
                self.camera_poses.append(best_pose)
            else:
                # Fall back to identity with offset
                pose = np.eye(4, dtype=np.float64)
                pose[2, 3] = -0.5 * i  # Simple translation along Z
                self.camera_poses.append(pose)

    def _triangulate_points(self):
        """Triangulate 3D points from matched features."""
        K = self.intrinsics.matrix()
        all_points = []
        all_colors = []

        # Triangulate from pairs of cameras
        for (i, j), matches in self.matches.items():
            if i >= len(self.camera_poses) or j >= len(self.camera_poses):
                continue

            feat_i = self.image_features[i]
            feat_j = self.image_features[j]

            # Get matched points
            pts1 = np.float64([feat_i.keypoints[m.queryIdx].pt for m in matches])
            pts2 = np.float64([feat_j.keypoints[m.trainIdx].pt for m in matches])

            # Projection matrices
            P1 = K @ self.camera_poses[i][:3]
            P2 = K @ self.camera_poses[j][:3]

            # Triangulate
            pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts3d = (pts4d[:3] / pts4d[3]).T

            # Filter points (remove outliers)
            valid = np.abs(pts3d).max(axis=1) < 100

            # Get colors from first image
            img = self.images[feat_i.image_idx]
            colors = []
            for pt, m in zip(pts3d, matches):
                kp = feat_i.keypoints[m.queryIdx].pt
                x, y = int(kp[0]), int(kp[1])
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    color = img[y, x][::-1] / 255.0  # BGR to RGB, normalize
                    colors.append(color)
                else:
                    colors.append([0.5, 0.5, 0.5])

            colors = np.array(colors)
            all_points.extend(pts3d[valid])
            all_colors.extend(colors[valid])

        if len(all_points) == 0:
            raise ValueError("Could not triangulate any points")

        self.points_3d = np.array(all_points)
        self.point_colors = np.array(all_colors)

        # Center and scale the point cloud
        centroid = np.median(self.points_3d, axis=0)
        self.points_3d -= centroid
        scale = np.percentile(np.abs(self.points_3d), 95)
        if scale > 0:
            self.points_3d /= scale

    def _dense_reconstruction(self):
        """Generate denser point cloud using patch-based stereo."""
        K = self.intrinsics.matrix()
        dense_points = list(self.points_3d)
        dense_colors = list(self.point_colors)

        # Use stereo matching for additional points (simplified)
        for idx in range(min(3, len(self.images) - 1)):
            img1 = self.images[idx]
            img2 = self.images[idx + 1]

            # Resize images to same size (use smaller dimensions)
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            target_h = min(h1, h2)
            target_w = min(w1, w2)

            if (h1, w1) != (target_h, target_w):
                img1 = cv2.resize(img1, (target_w, target_h))
            if (h2, w2) != (target_h, target_w):
                img2 = cv2.resize(img2, (target_w, target_h))

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Compute stereo disparity
            stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
            disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

            # Convert disparity to 3D points (sampled)
            h, w = disparity.shape
            step = 8  # Sample every 8 pixels

            for y in range(0, h, step):
                for x in range(0, w, step):
                    d = disparity[y, x]
                    if d > 1:  # Valid disparity
                        # Approximate 3D point
                        Z = (K[0, 0] * 0.1) / d  # baseline ~0.1
                        X = (x - K[0, 2]) * Z / K[0, 0]
                        Y = (y - K[1, 2]) * Z / K[1, 1]

                        if abs(X) < 5 and abs(Y) < 5 and 0 < Z < 10:
                            dense_points.append([X, Y, Z])
                            color = img1[y, x][::-1] / 255.0
                            dense_colors.append(color)

        self.points_3d = np.array(dense_points)
        self.point_colors = np.array(dense_colors)

        # Remove outliers using statistical filtering
        if len(self.points_3d) > 100:
            mean = np.mean(self.points_3d, axis=0)
            std = np.std(self.points_3d, axis=0)
            mask = np.all(np.abs(self.points_3d - mean) < 3 * std, axis=1)
            self.points_3d = self.points_3d[mask]
            self.point_colors = self.point_colors[mask]

    def _generate_mesh(self):
        """Generate a mesh from point cloud using ball pivoting or Poisson."""
        # Simple mesh generation using Delaunay triangulation on projected points
        # For better results, use Open3D or similar library

        if len(self.points_3d) < 4:
            return

        from scipy.spatial import Delaunay

        # Project to 2D for triangulation (XY plane)
        points_2d = self.points_3d[:, :2]

        try:
            tri = Delaunay(points_2d)
            self.triangles = tri.simplices

            # Filter long triangles (outliers)
            valid_triangles = []
            max_edge_length = 0.5  # Threshold

            for t in self.triangles:
                p0, p1, p2 = self.points_3d[t]
                e1 = np.linalg.norm(p1 - p0)
                e2 = np.linalg.norm(p2 - p1)
                e3 = np.linalg.norm(p0 - p2)

                if max(e1, e2, e3) < max_edge_length:
                    valid_triangles.append(t)

            self.triangles = np.array(valid_triangles) if valid_triangles else tri.simplices

        except Exception:
            # Fallback: create simple quad mesh
            self.triangles = None

    def _export_glb(self):
        """Export the model to GLB (binary glTF) format."""
        output_path = self.output_dir / "model.glb"

        # Build glTF structure
        gltf = self._build_gltf()

        # Write GLB
        self._write_glb(gltf, output_path)

    def _build_gltf(self) -> dict:
        """Build glTF JSON structure."""
        points = self.points_3d.astype(np.float32)
        colors = (self.point_colors * 255).astype(np.uint8)

        # Compute bounds
        min_pos = points.min(axis=0).tolist()
        max_pos = points.max(axis=0).tolist()

        # Binary buffer data
        position_data = points.tobytes()
        color_data = colors.tobytes()

        # Check if we have triangles for mesh, otherwise use point cloud
        has_mesh = hasattr(self, 'triangles') and self.triangles is not None and len(self.triangles) > 0

        if has_mesh:
            indices = self.triangles.astype(np.uint32)
            indices_data = indices.tobytes()
            buffer_data = position_data + color_data + indices_data
        else:
            buffer_data = position_data + color_data

        gltf = {
            "asset": {"version": "2.0", "generator": "Photogrammetry Pipeline"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "buffers": [{"byteLength": len(buffer_data)}],
            "bufferViews": [
                # Position buffer view
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(position_data),
                    "target": 34962  # ARRAY_BUFFER
                },
                # Color buffer view
                {
                    "buffer": 0,
                    "byteOffset": len(position_data),
                    "byteLength": len(color_data),
                    "target": 34962
                }
            ],
            "accessors": [
                # Position accessor
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(points),
                    "type": "VEC3",
                    "min": min_pos,
                    "max": max_pos
                },
                # Color accessor
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5121,  # UNSIGNED_BYTE
                    "normalized": True,
                    "count": len(colors),
                    "type": "VEC3"
                }
            ],
            "materials": [
                {
                    "pbrMetallicRoughness": {
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0
                    },
                    "doubleSided": True
                }
            ]
        }

        if has_mesh:
            # Add indices buffer view and accessor
            gltf["bufferViews"].append({
                "buffer": 0,
                "byteOffset": len(position_data) + len(color_data),
                "byteLength": len(indices_data),
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            })
            gltf["accessors"].append({
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": len(indices.flatten()),
                "type": "SCALAR"
            })
            gltf["meshes"] = [{
                "primitives": [{
                    "attributes": {"POSITION": 0, "COLOR_0": 1},
                    "indices": 2,
                    "material": 0,
                    "mode": 4  # TRIANGLES
                }]
            }]
        else:
            # Point cloud mode
            gltf["meshes"] = [{
                "primitives": [{
                    "attributes": {"POSITION": 0, "COLOR_0": 1},
                    "material": 0,
                    "mode": 0  # POINTS
                }]
            }]

        return {"json": gltf, "buffer": buffer_data}

    def _write_glb(self, gltf: dict, output_path: Path):
        """Write GLB binary file."""
        json_str = json.dumps(gltf["json"], separators=(',', ':'))

        # Pad JSON to 4-byte alignment
        while len(json_str) % 4 != 0:
            json_str += ' '

        json_bytes = json_str.encode('utf-8')
        bin_data = gltf["buffer"]

        # Pad binary to 4-byte alignment
        while len(bin_data) % 4 != 0:
            bin_data += b'\x00'

        # GLB header
        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

        with open(output_path, 'wb') as f:
            # Header
            f.write(b'glTF')  # magic
            f.write(struct.pack('<I', 2))  # version
            f.write(struct.pack('<I', total_length))  # length

            # JSON chunk
            f.write(struct.pack('<I', len(json_bytes)))  # chunk length
            f.write(b'JSON')  # chunk type
            f.write(json_bytes)

            # Binary chunk
            f.write(struct.pack('<I', len(bin_data)))  # chunk length
            f.write(b'BIN\x00')  # chunk type
            f.write(bin_data)

    def _generate_preview(self):
        """Generate a preview image of the 3D model."""
        preview_path = self.output_dir / "preview.png"

        # Simple preview: render point cloud from a fixed viewpoint
        img_size = 512
        preview = np.ones((img_size, img_size, 3), dtype=np.uint8) * 40  # Dark gray background

        if self.points_3d is None or len(self.points_3d) == 0:
            cv2.imwrite(str(preview_path), preview)
            return

        # Project points to 2D
        points = self.points_3d.copy()

        # Rotate for better view
        angle = np.pi / 6
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        Ry = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        points = (Ry @ Rx @ points.T).T

        # Project to image
        focal = 300
        cx, cy = img_size // 2, img_size // 2

        # Sort by depth for proper rendering
        z_order = np.argsort(-points[:, 2])

        for idx in z_order:
            pt = points[idx]
            if pt[2] < 0.1:
                continue

            x = int(focal * pt[0] / pt[2] + cx)
            y = int(focal * pt[1] / pt[2] + cy)

            if 0 <= x < img_size and 0 <= y < img_size:
                color = (self.point_colors[idx] * 255).astype(int)
                # Draw as small circle
                cv2.circle(preview, (x, y), 2, color.tolist(), -1)

        cv2.imwrite(str(preview_path), preview)

        # Also export point cloud as PLY for debugging
        self._export_ply()

    def _export_ply(self):
        """Export point cloud as PLY file."""
        ply_path = self.output_dir / "model.ply"

        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for pt, color in zip(self.points_3d, self.point_colors):
                r, g, b = (color * 255).astype(int)
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {r} {g} {b}\n")
