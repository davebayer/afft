function dst = fft2(src, m, n)
  arguments
    src
    m   = []
    n   = []
  end

  dst = afft_matlab(uint32(2001), src, m, n);
end
