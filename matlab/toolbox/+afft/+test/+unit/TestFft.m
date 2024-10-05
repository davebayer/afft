classdef TestFft < afft.test.unit.AbstractTestTransform
  properties (Constant, TestParameter)
    srcComplexity = {'complex'}; % todo: add 'real' when implemented
    dim           = {1}; % todo" add 1:afft.maxDimCount when implemented
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

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, srcComplexity, dim)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, srcComplexity);

      dstRef = afft.test.unit.TestFft.computeReference(src, dim, normalization);
      dst    = afft.fft(src, ...
                        [], ...
                        dim, ...
                        'backend',       backend, ...
                        'normalization', normalization, ...
                        'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, srcComplexity, dim)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, srcComplexity);

      try
        dst = afft.fft(src, ...
                       [], ...
                       dim, ...
                       'backend',       backend, ...
                       'normalization', normalization, ...
                       'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        testCase.verifyFail('Expected afft.fft to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, srcComplexity, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('cufft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0) || (dim ~= 1 && dim ~= ndims(gridSize))
        testFailure(testCase, 'cufft', precision, normalization, gridSize, srcComplexity, dim);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize, srcComplexity, dim);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize, srcComplexity, dim)
      testCase.assumeTrue(afft.hasBackend('fftw3'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, srcComplexity, dim);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, srcComplexity, dim);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, srcComplexity, dim)
      testCase.assumeTrue(afft.hasBackend('mkl'));

      if dim ~= 1 && dim ~= ndims(gridSize)
        testFailure(testCase, 'mkl', precision, normalization, gridSize, srcComplexity, dim);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize, srcComplexity, dim);
      end
    end

    function testPocketfft(testCase, precision, normalization, gridSize, srcComplexity, dim)
      testCase.assumeTrue(afft.hasBackend('pocketfft'));

      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, srcComplexity, dim);
    end

    function testVkfft(testCase, precision, normalization, gridSize, srcComplexity, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('vkfft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, srcComplexity, dim);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, srcComplexity, dim);
      end
    end
  end
end
