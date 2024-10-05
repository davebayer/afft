classdef TestFft2 < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    srcComplexity = {'complex'}; % todo: add 'real' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, normalization)
      % Compute the reference using the built-in fft2 function.
      dstRef = fft2(src);

      transformSize = size(src, 1) * size(src, 2);

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
    function testSuccess(testCase, backend, precision, normalization, gridSize, srcComplexity)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, srcComplexity);

      dstRef = afft.test.unit.TestFft2.computeReference(src, normalization);
      dst    = afft.fft2(src, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, srcComplexity)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, srcComplexity);

      try
        dst = afft.fft2(src, ...
                        'backend',       backend, ...
                        'normalization', normalization, ...
                        'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        testCase.verifyFail('Expected afft.fft2 to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, srcComplexity)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('cufft'));

      if not(strcmp(normalization, 'none')) && sum(gridSize) > 0
        testFailure(testCase, 'cufft', precision, normalization, gridSize, srcComplexity);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize, srcComplexity);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize, srcComplexity)
      testCase.assumeTrue(afft.hasBackend('fftw3'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, srcComplexity);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, srcComplexity);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, srcComplexity)
      testCase.assumeTrue(afft.hasBackend('mkl'));

      testSuccess(testCase, 'mkl', precision, normalization, gridSize, srcComplexity);
    end

    function testPocketfft(testCase, precision, normalization, gridSize, srcComplexity)
      testCase.assumeTrue(afft.hasBackend('pocketfft'));

      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, srcComplexity);
    end

    function testVkfft(testCase, precision, normalization, gridSize, srcComplexity)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('vkfft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, srcComplexity);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, srcComplexity);
      end
    end
  end
end
