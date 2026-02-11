import os
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================== 配置区 ==================
VIDEO_DIR = "/media/zjq/6CDE775EDE771F8E/lxk/NTU-data/nturgbd_raw"
OUT_DIR = os.path.join(VIDEO_DIR, "fall_rgb")
EXT = "_rgb.avi"
NUM_WORKERS = max(cpu_count() - 2, 1)  # 留点 CPU 给系统
# ===========================================

os.makedirs(OUT_DIR, exist_ok=True)

def extract_one(video_name):
    video_path = os.path.join(VIDEO_DIR, video_name)
    base = video_name.replace(EXT, "")
    out_dir = os.path.join(OUT_DIR, base)

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        return f"[SKIP] {base}"

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", video_path,
        "-q:v", "2",
        os.path.join(out_dir, "img_%05d.jpg")
    ]

    try:
        subprocess.run(cmd, check=True)
        return f"[OK] {base}"
    except subprocess.CalledProcessError:
        return f"[FAIL] {base}"

def main():
    videos = sorted([v for v in os.listdir(VIDEO_DIR) if v.endswith(EXT)])
    print(f"Found {len(videos)} videos")
    print(f"Using {NUM_WORKERS} processes")

    with Pool(NUM_WORKERS) as pool:
        for msg in tqdm(pool.imap_unordered(extract_one, videos), total=len(videos)):
            pass

if __name__ == "__main__":
    main()
