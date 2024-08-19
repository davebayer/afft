function dst = fft(src, size, axis)
  arguments
    src
    size = []
    axis = [1]
  end

  dst = afft(uint32(2000), src, size, axis);
end
