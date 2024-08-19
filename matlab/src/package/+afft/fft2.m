function dst = fft2(src, m, n)
  arguments
    src
    m   = []
    n   = []
  end

  dst = afft(uint32(2001), src, m, n);
end
