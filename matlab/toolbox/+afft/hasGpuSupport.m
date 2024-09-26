function result = hasGpuSupport()
  result = afft.internal.afft_matlab(uint32(1));
end
