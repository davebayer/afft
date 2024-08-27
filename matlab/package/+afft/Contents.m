% afft library for computing fft-related transformations
% Version 0.0.1 27-Aug-2024
%
% Package management functions
%   clearPlanCache          - clear the plan cache releasing all created plans
% 
% Standalone Plan class
%   Plan                    - class for creating and executing transform plans
%
% Immediate transform functions (equvalents to original matlab functions)
%   Discrete Fourier Transform (DFT)
%     fft                   - compute the 1D DFT
%     fft2                  - compute the 2D DFT
%     fftn                  - compute the N-D DFT
%     ifft                  - compute the 1D IDFT
%     ifft2                 - compute the 2D IDFT
%     ifftn                 - compute the N-D IDFT
%   Discrete Hartley Transform (DHT)
%     dht                   - compute the 1D DHT
%     dht2                  - compute the 2D DHT
%     dhtn                  - compute the N-D DHT
%     idht                  - compute the 1D IDHT
%     idht2                 - compute the 2D IDHT
%     idhtn                 - compute the N-D IDHT
%   Discrete Cosine Transform (DCT)
%     dct                   - compute the 1D DCT
%     dct2                  - compute the 2D DCT
%     dctn                  - compute the N-D DCT
%     idct                  - compute the 1D IDCT
%     idct2                 - compute the 2D IDCT
%     idctn                 - compute the N-D IDCT
%   Discrete Sine Transform (DST)
%     dst                   - compute the 1D DST
%     dst2                  - compute the 2D DST
%     dstn                  - compute the N-D DST
%     idst                  - compute the 1D IDST
%     idst2                 - compute the 2D IDST
%     idstn                 - compute the N-D IDST
%   Discrete Trigonomic Transform (DTT)
%     dtt                   - compute the 1D DTT
%     dtt2                  - compute the 2D DTT
%     dttn                  - compute the N-D DTT
%     idtt                  - compute the 1D IDTT
%     idtt2                 - compute the 2D IDTT
%     idttn                 - compute the N-D IDTT
% 
% See also fft, fft2, fftn, ifft, ifft2, ifftn, dct, dct2, idct, idct2, dst, idst
%
% This file is part of the afft library. For more information, see the official <a href="matlab:
% web('https://github.com/DejvBayer/afft.git')">afft GitHub</a>.
