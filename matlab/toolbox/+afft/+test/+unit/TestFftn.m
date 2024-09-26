classdef TestFftn < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    srcComplexity = {'complex'}; % todo: add 'real' when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, normalization)
      % Compute the reference using the built-in fftn function.
      dstRef = fftn(src);

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
      elseif strcmp(normalization, 'unitary')
        dstRef = dstRef / numel(src);
      elseif strcmp(normalization, 'orthogonal')
        dstRef = dstRef / sqrt(numel(src));
      else
        error('Invalid normalization');
      end
    end
  end

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, srcComplexity)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, srcComplexity);

      dstRef = afft.test.unit.TestFftn.computeReference(src, normalization);
      dst    = afft.fftn(src, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, srcComplexity)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, srcComplexity);

      try
        dst = afft.fftn(src, ...
                        'backend',       backend, ...
                        'normalization', normalization, ...
                        'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        testCase.verifyFail('Expected afft.fftn to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, srcComplexity)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('cufft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0) || numel(gridSize) > 3
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

      if numel(gridSize) > 7
        testFailure(testCase, 'mkl', precision, normalization, gridSize, srcComplexity);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize, srcComplexity);
      end
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
