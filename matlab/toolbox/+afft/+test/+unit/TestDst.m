classdef TestDst < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    dstType = {[], 1, 2, 3, 4}; 
    dim     = {1}; % todo" add 1:afft.maxDimCount when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, dim, normalization, dstType)
      if dstType == 1
        transformSize = 2 * (size(src, dim) - 1);
      else
        transformSize = 2 * size(src, dim);
      end

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
        if isempty(dstType)
          dstRef = matlab.internal.math.transform.mldst(gather(src), [], dim);
        else
          dstRef = matlab.internal.math.transform.mldst(gather(src), [], dim, 'Variant', dstType);
        end
      elseif strcmp(normalization, 'unitary')
        if isempty(dstType)
          dstRef = matlab.internal.math.transform.mldst(gather(src), [], dim) / transformSize;
        else
          dstRef = matlab.internal.math.transform.mldst(gather(src), [], dim, 'Variant', dstType) / transformSize;
        end
      elseif strcmp(normalization, 'orthogonal')
        if isempty(dstType)
          dstRef = dst(src, [], dim);
        else
          dstRef = dst(src, [], dim, Type=dstType);
        end
      else
        error('Invalid normalization');
      end
    end
  end

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, dstType, dim)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      dstRef = afft.test.unit.TestDst.computeReference(src, dim, normalization, dstType);

      if isempty(dstType)
        dst = afft.dst(src, ...
                       [], ...
                       dim, ...
                       'backend',       backend, ...
                       'normalization', normalization, ...
                       'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
      else
        dst = afft.dst(src, ...
                       [], ...
                       dim, ...
                       'type',          dstType, ...
                       'backend',       backend, ...
                       'normalization', normalization, ...
                       'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
      end

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, dstType, dim)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      try
        if isempty(dstType)
          dst = afft.dst(src, ...
                         [], ...
                         dim, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        else
          dst = afft.dst(src, ...
                         [], ...
                         dim, ...
                         'type',          dstType, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        end
        testCase.verifyFail('Expected afft.dst to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, dstType, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('cufft'));

      if sum(gridSize) > 0
        testFailure(testCase, 'cufft', precision, normalization, gridSize, dstType, dim);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize, dstType, dim);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize, dstType, dim)
      testCase.assumeTrue(afft.hasBackend('fftw3'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, dstType, dim);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, dstType, dim);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, dstType, dim)
      testCase.assumeTrue(afft.hasBackend('mkl'));

      if sum(gridSize) > 0
        testFailure(testCase, 'mkl', precision, normalization, gridSize, dstType, dim);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize, dstType, dim);
      end
    end

    function testPocketfft(testCase, precision, normalization, gridSize, dstType, dim)
      testCase.assumeTrue(afft.hasBackend('pocketfft'));

      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, dstType, dim);
    end

    function testVkfft(testCase, precision, normalization, gridSize, dstType, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('vkfft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, dstType, dim);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, dstType, dim);
      end
    end
  end
end
