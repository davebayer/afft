function dst = fftn(src, size)
  arguments
    src
    size = []
  end

  dst = afft_matlab(uint32(2002), src, size);
end
