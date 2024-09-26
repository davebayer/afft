function result = hasBackend(backend)
  result = afft.internal.afft_matlab(uint32(2), backend);
end
