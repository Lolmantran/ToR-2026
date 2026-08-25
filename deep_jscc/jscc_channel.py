import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20):
        if channel_type not in ['AWGN', 'Rayleigh']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')
        
        # if z_hat.dim() == 4:
        #     # k = np.prod(z_hat.size()[1:])
        #     k = torch.prod(torch.tensor(z_hat.size()[1:]))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        # elif z_hat.dim() == 3:
        #     # k = np.prod(z_hat.size())
        #     k = torch.prod(torch.tensor(z_hat.size()))
        #     sig_pwr = torch.sum(torch.abs(z_hat).square()) / k
            
        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
        
        # Matches the reference implementation (chunbaobao/Deep-JSCC-PyTorch)
        # exactly, including the /2 real/imag split. Combined with the
        # matching normalization convention above, this reproduces the
        # reference repo's own ~30 dB benchmark results (see jscc_model.py
        # docstring for the documented +3.01 dB offset vs the paper's literal
        # Eq. 1 -- intentionally kept for comparability with the reference repo).
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr/2)
        if self.channel_type == 'Rayleigh':
            # hc = torch.randn_like(z_hat)  wrong implement before
            # hc = torch.randn(1, device = z_hat.device) 
            hc = torch.randn(2, device = z_hat.device) 
        
            # clone for in-place operation  
            z_hat = z_hat.clone()
            z_hat[:,:z_hat.size(1)//2] = hc[0] * z_hat[:,:z_hat.size(1)//2]
            z_hat[:,z_hat.size(1)//2:] = hc[1] * z_hat[:,z_hat.size(1)//2:]
            

            # z_hat = hc * z_hat

        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == '__main__':
    # test
    channel = Channel(channel_type='AWGN', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)

    channel = Channel(channel_type='Rayleigh', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)
