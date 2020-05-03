import torch
def change_ckpt_iter(ckpt_path, save_path, i):
    c = torch.load(ckpt_path)
    c['iteration'] = i
    torch.save(c,save_path)

def check_ckpt_iter(ckpt_path):
    c = torch.load(ckpt_path)
    print(c['iteration'])

if __name__=='__main__':
    change_ckpt_iter('models/control/O0/model_0007955.pth', 'models/control/O0/best.pth', 0)
