import os
import glob
import shutil
import threading
import time
from typing import Tuple

def start_pointcloud_watcher(
    point_cloud_dir: str, destination_dir: str
) -> Tuple[threading.Event, threading.Thread]:
    """Start a watcher thread to keep only the latest point_cloud.ply file in 'current' and delete old iteration folders.
    Args:
        point_cloud_dir (str): Path to the point cloud directory (should contain iteration_* subdirs).
    Returns:
        stop_flag (threading.Event): Event to signal the watcher to stop.
        watcher_thread (threading.Thread): The watcher thread instance.
    """
    stop_flag = threading.Event()
    last_seen_ply = None
    last_seen_mtime = None

    def watcher():
        nonlocal last_seen_ply, last_seen_mtime
        while not stop_flag.is_set():
            iteration_dirs = sorted(
                glob.glob(os.path.join(point_cloud_dir, "iteration_*")),
                key=lambda d: os.path.getmtime(d),
            )
            if iteration_dirs:
                latest_dir = iteration_dirs[-1]
                # Delete all but the latest iteration dir
                for it_dir in iteration_dirs[:-1]:
                    shutil.rmtree(it_dir, ignore_errors=True)
                ply_src = os.path.join(latest_dir, "point_cloud.ply")
                ply_dst = os.path.join(destination_dir, "point_cloud.ply")
                if os.path.exists(ply_src):
                    mtime = os.path.getmtime(ply_src)
                    # Only copy if the file is new or has changed
                    if last_seen_ply != ply_src or last_seen_mtime != mtime:
                        last_seen_ply = ply_src
                        last_seen_mtime = mtime
                        # Copy to destination_dir only
                        os.makedirs(destination_dir, exist_ok=True)
                        if os.path.exists(ply_dst):
                            os.remove(ply_dst)
                        shutil.copy2(ply_src, ply_dst)
            time.sleep(2)

    watcher_thread = threading.Thread(target=watcher, daemon=True)
    watcher_thread.start()
    return stop_flag, watcher_thread
