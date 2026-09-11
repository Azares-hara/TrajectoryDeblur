from lossfunc.lpips import LPIPS

_lpips_instance = None

def get_lpips(net="squeeze", device="cuda:0", use_half=False):
    global _lpips_instance
    if _lpips_instance is None:
        _lpips_instance = LPIPS(net=net, device=device, use_half=use_half)
    return _lpips_instance

