import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md","LICENSE",".git"],
    pythonpath=True,
    #dotenv=True,
)

from pathlib import Path
import glob

def find_latest_checkpoints(ckpt_dir):
    ckpts = sorted(glob.glob(str(Path(ckpt_dir / "*.ckpt"))))
    if len(ckpts) == 0:
        return None
    else:
        return ckpts[-1]

if __name__ == '__main__':
    ckpt_dir = root / "lightning_logs/*/checkpoints"
    #print(ckpt_dir)
    resume_ckpt = find_latest_checkpoints(ckpt_dir)
    # Todo: パスの指定のさせ方を考える
    print(resume_ckpt)