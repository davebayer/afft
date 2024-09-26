function result = hasGpuSupport()
  result = internal.afft_matlab(uint32(1));
end
