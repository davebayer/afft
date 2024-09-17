% afft library for computing fft-related transformations
% Version 0.0.1 27-Aug-2024
%
% Toolbox management functions
%   clearPlanCache          - clear the plan cache releasing all created plans
% 
% Standalone Plan class
%   Plan                    - class for creating and executing transform plans
%   
% Immediate transform functions (equvalents to original matlab functions)
%   Discrete Fourier Transform (DFT)
%     fft                   - compute the forward 1D DFT
%     fft2                  - compute the forward 2D DFT
%     fftn                  - compute the forward ND DFT
%     ifft                  - compute the inverse 1D DFT
%     ifft2                 - compute the inverse 2D DFT
%     ifftn                 - compute the inverse ND DFT
%     rfft                  - compute the forward 1D real DFT
%     rfft2                 - compute the forward 2D real DFT
%     rfftn                 - compute the forward ND real DFT
%     irfft                 - compute the inverse 1D real DFT
%     irfft2                - compute the inverse 2D real DFT
%     irfftn                - compute the inverse ND real DFT
%   Discrete Hartley Transform (DHT)
%     dht                   - compute the forward 1D DHT
%     dht2                  - compute the forward 2D DHT
%     dhtn                  - compute the forward ND DHT
%     idht                  - compute the inverse 1D DHT
%     idht2                 - compute the inverse 2D DHT
%     idhtn                 - compute the inverse ND DHT
%   Discrete Cosine Transform (DCT)
%     dct                   - compute the 1D DCT
%     dct2                  - compute the 2D DCT
%     dctn                  - compute the ND DCT
%     idct                  - compute the 1D DCT
%     idct2                 - compute the 2D DCT
%     idctn                 - compute the ND DCT
%   Discrete Sine Transform (DST)
%     dst                   - compute the forward 1D DST
%     dst2                  - compute the forward 2D DST
%     dstn                  - compute the forward ND DST
%     idst                  - compute the inverse 1D DST
%     idst2                 - compute the inverse 2D DST
%     idstn                 - compute the inverse ND DST
%   Discrete Trigonomic Transform (DTT)
%     dtt                   - compute the forward 1D DTT
%     dtt2                  - compute the forward 2D DTT
%     dttn                  - compute the forward ND DTT
%     idtt                  - compute the inverse 1D DTT
%     idtt2                 - compute the inverse 2D DTT
%     idttn                 - compute the inverse ND DTT
% 
% See also fft, fft2, fftn, ifft, ifft2, ifftn, dct, dct2, idct, idct2, dst, idst
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">GitHub</a>.
