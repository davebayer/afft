classdef TestDct < afft.test.unit.AbstractTestTransform
  properties (TestParameter)
    dctType = {[], 1, 2, 3, 4}; 
    dim     = {1}; % todo" add 1:afft.maxDimCount when implemented
  end

  methods (Static)
    function dstRef = computeReference(src, dim, normalization, dctType)
      if dctType == 1
        transformSize = 2 * (size(src, dim) - 1);
      else
        transformSize = 2 * size(src, dim);
      end

      % Modify the reference to match the given normalization.
      if strcmp(normalization, 'none')
        if isempty(dctType)
          dstRef = matlab.internal.math.transform.mldct(gather(src), [], dim);
        else
          dstRef = matlab.internal.math.transform.mldct(gather(src), [], dim, 'Variant', dctType);
        end
      elseif strcmp(normalization, 'unitary')
        if isempty(dctType)
          dstRef = matlab.internal.math.transform.mldct(gather(src), [], dim) / transformSize;
        else
          dstRef = matlab.internal.math.transform.mldct(gather(src), [], dim, 'Variant', dctType) / transformSize;
        end
      elseif strcmp(normalization, 'orthogonal')
        if isempty(dctType)
          dstRef = dct(src, [], dim);
        else
          dstRef = dct(src, [], dim, Type=dctType);
        end
      else
        error('Invalid normalization');
      end
    end
  end

  methods
    function testSuccess(testCase, backend, precision, normalization, gridSize, dctType, dim)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      dstRef = afft.test.unit.TestDct.computeReference(src, dim, normalization, dctType);

      if isempty(dctType)
        dst = afft.dct(src, ...
                       [], ...
                       dim, ...
                       'backend',       backend, ...
                       'normalization', normalization, ...
                       'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
      else
        dst = afft.dct(src, ...
                       [], ...
                       dim, ...
                       'type',          dctType, ...
                       'backend',       backend, ...
                       'normalization', normalization, ...
                       'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
      end

      compareResults(testCase, precision, dstRef, dst);
    end

    function testFailure(testCase, backend, precision, normalization, gridSize, dctType, dim)
      src = afft.test.unit.AbstractTestTransform.generateSrcArray(backend, gridSize, precision, 'real');

      try
        if isempty(dctType)
          dst = afft.dct(src, ...
                         [], ...
                         dim, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        else
          dst = afft.dct(src, ...
                         [], ...
                         dim, ...
                         'type',          dctType, ...
                         'backend',       backend, ...
                         'normalization', normalization, ...
                         'threadLimit',   afft.test.unit.AbstractTestTransform.cpuThreadLimit);
        end
        testCase.verifyFail('Expected afft.dct to fail');
      catch
      end
    end
  end

  methods (Test)
    function testCufft(testCase, precision, normalization, gridSize, dctType, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('cufft'));

      if sum(gridSize) > 0
        testFailure(testCase, 'cufft', precision, normalization, gridSize, dctType, dim);
      else
        testSuccess(testCase, 'cufft', precision, normalization, gridSize, dctType, dim);
      end
    end

    function testFftw3(testCase, precision, normalization, gridSize, dctType, dim)
      testCase.assumeTrue(afft.hasBackend('fftw3'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'fftw3', precision, normalization, gridSize, dctType, dim);
      else
        testSuccess(testCase, 'fftw3', precision, normalization, gridSize, dctType, dim);
      end
    end

    function testMkl(testCase, precision, normalization, gridSize, dctType, dim)
      testCase.assumeTrue(afft.hasBackend('mkl'));

      if sum(gridSize) > 0
        testFailure(testCase, 'mkl', precision, normalization, gridSize, dctType, dim);
      else
        testSuccess(testCase, 'mkl', precision, normalization, gridSize, dctType, dim);
      end
    end

    function testPocketfft(testCase, precision, normalization, gridSize, dctType, dim)
      testCase.assumeTrue(afft.hasBackend('pocketfft'));

      testSuccess(testCase, 'pocketfft', precision, normalization, gridSize, dctType, dim);
    end

    function testVkfft(testCase, precision, normalization, gridSize, dctType, dim)
      testCase.assumeTrue(afft.hasGpuSupport && afft.hasBackend('vkfft'));

      if (not(strcmp(normalization, 'none')) && sum(gridSize) > 0)
        testFailure(testCase, 'vkfft', precision, normalization, gridSize, dctType, dim);
      else
        testSuccess(testCase, 'vkfft', precision, normalization, gridSize, dctType, dim);
      end
    end
  end
end
