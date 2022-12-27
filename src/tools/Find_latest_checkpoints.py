from pathlib import Path
import glob
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md", "LICENSE", ".git"],
    pythonpath=True,
    # dotenv=True,
)


def find_latest_checkpoints(ckpt_dir: Path):
    ckpts = sorted(glob.glob(str(Path(ckpt_dir / "*.ckpt"))))
    if len(ckpts) == 0:
        return None
    else:
        return ckpts[-1]  # ,ckpts


if __name__ == "__main__":
    ckpt_dir = root / "lightning_logs/*/checkpoints"
    # print(ckpt_dir)
    # resume_ckpt, ckpts = find_latest_checkpoints(ckpt_dir)
    resume_ckpt = find_latest_checkpoints(ckpt_dir)
    # Todo: パスの指定のさせ方を考える
    print(resume_ckpt)
    # print(ckpts)
