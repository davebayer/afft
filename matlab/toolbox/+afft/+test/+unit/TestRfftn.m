classdef TestRfftn < afft.test.unit.AbstractTestTransform
  methods (Static)
    function dstRef = computeReference(src, normalization)
      % Compute the reference using the built-in fftn function.
      dstRef = fftn(src);

      if not(isempty(dstRef))
        % Reduce the x dimension to (x / 2 + 1)
        xReduced = uint64(fix(size(dstRef, 1) / 2)) + 1;
        dstRef   = dstRef(1:xReduced, :, :);
      end

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
    function testSuccess(testCase, backend, precision, normalization, gridSize)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      dstRef = afft.test.unit.TestRfftn.computeReference(src, normalization);
      dst    = afft.rfftn(src, ...
                          'backend',       backend, ...
                          'normalization', normalization, ...
                          'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      try
        dst = afft.rfftn(src, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        testCase.verifyFail('Expected afft.rfftn to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize)
      testCase.assumeTrue(afft.hasGpuSupport)

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0) || numel(gridSize) > 3
        testFailure(testCase, 'cufft', precision, normalization, gridSize);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize)
      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize)
      if numel(gridSize) > 7
        testFailure(testCase, 'mkl', precision, normalization, gridSize);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize);
      end
    end

    function testPocketfft(testCase, precision, normalization, gridSize)
      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize);
    end

    function testVkfft(testCase, precision, normalization, gridSize)
      testCase.assumeTrue(afft.hasGpuSupport)
      
      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize);
      end
    end
  end
end
