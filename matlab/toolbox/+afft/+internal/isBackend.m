function res = isBackend(str)
  res = ischar(str) && ismember(str, {'cufft', 'fftw3', 'mkl', 'pocketfft', 'vkfft'});
end
