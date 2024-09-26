classdef TestFft < AbstractTestTransform
  properties (Constant, TestParameter)
    srcComplexity = {'complex'}; % todo: add 'real' when implemented
    dim           = {1};
  end

  methods (Static)
    function dstRef = computeReference(src, dim, normalization)
      % Compute the reference using the built-in fft2 function.
      dstRef = fft(src, [], dim);

      transformSize = size(src, dim);

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
      elseif strcmp(normalization, 'unitary')
        dstRef = dstRef / transformSize;
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef / sqrt(transformSize);
      else
        error('Invalid normalization');
      end
    end
  end

  methods (Test)
    function testCpu(testCase, precision, srcComplexity, normalization, gridSize, dim)
      src = AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity);

      dstRef = TestFft.computeReference(src, dim, normalization);
      dst    = afft.fft(src, [], dim, 'normalization', normalization);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testGpu(testCase, precision, srcComplexity, normalization, gridSize, dim)
      src = gpuArray(AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity));

      dstRef = TestFft.computeReference(src, dim, normalization);
      dst    = afft.fft(src, [], dim, 'normalization', normalization);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testCufft(testCase, precision, srcComplexity, normalization, gridSize, dim)
      src = gpuArray(AbstractTestTransform.generateSrcArray(gridSize, precision, srcComplexity));

      dstRef = TestFft.computeReference(src, dim, normalization);



      dst = afft.fft(src, [], dim, 'normalization', normalization);

      compareResults(testCase, precision, dstRef, dst);
    end
  end
end