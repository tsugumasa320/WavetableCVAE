import glob
def find_latest_checkpoints(checkpoint_dir):
    ckpts = sorted(glob.glob(checkpoint_dir+"/*.ckpt"))
    if len(ckpts) == 0:
        return None
    else:
        return ckpts[-1]

checkpoint_dir = "lightning_logs/*/checkpoints"
resume_ckpt = find_latest_checkpoints(checkpoint_dir)

if __name__ == '__main__':
    find_latest_checkpoints(checkpoint_dir)
    # Todo: パスの指定のさせ方を考える
    print(resume_ckpt)