/*
  This file is part of afft library.

  Copyright (c) 2024 David Bayer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef AFFT_DETAIL_CUDA_RTC_RTC_HPP
#define AFFT_DETAIL_CUDA_RTC_RTC_HPP

#ifndef AFFT_TOP_LEVEL_INCLUDE
# include "../../include.hpp"
#endif

#include "error.hpp"
#include "../enviroment.hpp"

namespace afft::detail::cuda::rtc
{
  /**
   * @struct CSymbolName
   * @brief A wrapper around a C symbol name.
   */
  struct CSymbolName : public std::string_view
  {
    /**
     * @brief Make all constructors explicit.
     * @tparam Args Argument types.
     * @param args Arguments to forward to the std::string_view constructor.
     */
    template<typename... Args>
    explicit constexpr CSymbolName(Args&&... args)
    : std::string_view(std::forward<Args>(args)...)
    {}

    /// @brief Make all operators explicit.
    using std::string_view::operator=;
    using std::string_view::operator[];
  };

  /**
   * @struct CppSymbolName
   * @brief A wrapper around a C++ symbol name.
   */
  struct CppSymbolName : public std::string_view
  {
    /**
     * @brief Make all constructors explicit.
     * @tparam Args Argument types.
     * @param args Arguments to forward to the std::string_view constructor.
     */
    template<typename... Args>
    explicit constexpr CppSymbolName(Args&&... args)
    : std::string_view(std::forward<Args>(args)...)
    {}

    /// @brief Make all operators explicit.
    using std::string_view::operator=;
    using std::string_view::operator[];
  };

  /**
   * @class CppLoweredSymbolName
   * @brief A wrapper around a C++ lowered symbol name. This class can be created only by the Program class.
   */
  class CppLoweredSymbolName : public std::string_view
  {
    public:
      /**
       * @class PriviledgeToken
       * @brief A token to allow the creation of the CppLoweredSymbolName class.
       */
      class PriviledgeToken
      {
        friend class Program;
      };

      /// @brief Default constructor.
      CppLoweredSymbolName() = default;

      /**
       * @brief Constructor.
       * @param name The name of the symbol.
       * @param program The program that contains the symbol.
       * @param token The privilidge token.
       */
      CppLoweredSymbolName(std::string_view                                     name,
                           std::shared_ptr<std::remove_pointer_t<nvrtcProgram>> program,
                           PriviledgeToken)
      : std::string_view(name), mProgram(std::move(program))
      {}

      /// @brief Copy constructor.
      CppLoweredSymbolName(const CppLoweredSymbolName&) = default;

      /// @brief Move constructor.
      CppLoweredSymbolName(CppLoweredSymbolName&&) = default;

      /// @brief Destructor.
      ~CppLoweredSymbolName() = default;

      /// @brief Copy assignment operator.
      CppLoweredSymbolName& operator=(const CppLoweredSymbolName&) = default;

      /// @brief Move assignment operator.
      CppLoweredSymbolName& operator=(CppLoweredSymbolName&&) = default;
    private:
      std::shared_ptr<std::remove_pointer_t<nvrtcProgram>> mProgram{}; ///< The program that contains the symbol.
  };

  /// @brief NVRTC code types.
  enum class CodeType
  {
    PTX,     ///< PTX code.
    CUBIN,   ///< CUBIN code.
    LTOIR,   ///< LTOIR code.
    OptixIR, ///< OptiX IR code.
  };

  /**
   * @class Code
   * @brief Program generated code.
   */
  class Code
  {
    public:
      /**
       * @class PrivilegedToken
       * @brief A token to allow the creation of the Code class.
       */
      class PrivilegedToken
      {
        friend class Program;
      };

      /// @brief Default constructor.
      Code() = delete;

      /**
       * @brief Constructor.
       * @param type The code type.
       * @param size The code size.
       */
      Code(CodeType type, std::size_t size, PrivilegedToken)
      : mType(type), mCode(size, '\0')
      {}

      /// @brief Copy constructor.
      Code(const Code&) = default;

      /// @brief Move constructor.
      Code(Code&&) = default;

      /// @brief Destructor.
      ~Code() = default;

      /// @brief Copy assignment operator.
      Code& operator=(const Code&) = default;

      /// @brief Move assignment operator.
      Code& operator=(Code&&) = default;

      /// @brief Get the code type.
      [[nodiscard]] CodeType type() const
      {
        return mType;
      }

      /// @brief Get the code data.
      [[nodiscard]] char* data()
      {
        return mCode.data();
      }

      /// @brief Get the code data.
      [[nodiscard]] const char* data() const
      {
        return mCode.data();
      }

      /// @brief Get the code size.
      [[nodiscard]] std::size_t size() const
      {
        return mCode.size();
      }
    protected:
    private:
      CodeType    mType{}; ///< The code type.
      std::string mCode{}; ///< The code.
  };

  /**
   * @struct Header
   * @brief A header for the program.
   */
  struct Header
  {
    std::string_view srcCode{};     ///< The source code of the header.
    std::string_view includeName{}; ///< The include name of the header.
  };

  /**
   * @class Program
   * @brief A program for the CUDA runtime compilation.
   */
  class Program
  {
    public:
      /// @brief Default constructor.
      Program() = delete;

      /**
       * @brief Constructor.
       * @tparam n The number of headers.
       * @param srcCode The source code of the program.
       * @param programName The name of the program.
       * @param headers The headers of the program.
       */
      template<std::size_t n = 0>
      Program(std::string_view srcCode, std::string_view programName, Span<Header, n> headers = {})
      {
        using CharPtrContainer = std::conditional_t<(n != dynamicExtent),
                                                    std::array<const char*, n>, std::vector<const char*>>;

        CharPtrContainer headerSrcCodePtrs{};
        CharPtrContainer headerIncludeNamePtrs{};

        if constexpr (n == dynamicExtent)
        {
          headerSrcCodePtrs.resize(headers.size());
          headerIncludeNamePtrs.resize(headers.size());
        }

        for (std::size_t i{}; i < headers.size(); ++i)
        {
          headerSrcCodePtrs[i]     = headers[i].srcCode.data();
          headerIncludeNamePtrs[i] = headers[i].includeName.data();
        }

        nvrtcProgram program{};

        Error::check(nvrtcCreateProgram(&program,
                                        srcCode.data(),
                                        programName.data(),
                                        static_cast<int>(headers.size()),
                                        headerSrcCodePtrs.data(),
                                        headerIncludeNamePtrs.data()));

        mProgram.reset(program, Deleter{});
      }

      /// @brief Copy constructor.
      Program(const Program&) = delete;

      /// @brief Move constructor.
      Program(Program&&) = default;

      /// @brief Destructor.
      ~Program() = default;

      /// @brief Copy assignment operator.
      Program& operator=(const Program&) = delete;

      /// @brief Move assignment operator.
      Program& operator=(Program&&) = default;

      /**
       * @brief Add a name expression.
       * @param cSymbolName The name of the symbol.
       */
      void addNameExpression(CSymbolName cSymbolName)
      {
        addNameExpression(std::string_view{cSymbolName});
      }

      /**
       * @brief Add a name expression.
       * @param cppSymbolName The name of the symbol.
       */
      void addNameExpression(CppSymbolName cppSymbolName)
      {
        addNameExpression(std::string_view{cppSymbolName});
      }

      /**
       * @brief Compile the program.
       * @param options The compilation options.
       * @return A tuple containing a boolean indicating if the compilation was successful and a string with the log.
       */
      [[nodiscard]] bool compile(Span<const char*> options)
      {
        if (std::exchange(mIsCompiled, true))
        {
          throw makeException<std::runtime_error>("The program is already compiled");
        }

        bool ok = Error::isOk(nvrtcCompileProgram(mProgram.get(), static_cast<int>(options.size()), options.data()));

        std::size_t logSize{};

        Error::check(nvrtcGetProgramLogSize(mProgram.get(), &logSize));

        mCompilationLog.resize(logSize);

        Error::check(nvrtcGetProgramLog(mProgram.get(), mCompilationLog.data()));

        return ok;
      }

      /**
       * @brief Get the lowered symbol name.
       * @param name The name of the symbol.
       * @return The lowered symbol name.
       */
      [[nodiscard]] CppLoweredSymbolName getLoweredSymbolName(CppSymbolName cppSymbolName)
      {
        if (!mIsCompiled)
        {
          throw makeException<std::runtime_error>("The program is not compiled");
        }

        const char* loweredName{};

        Error::check(nvrtcGetLoweredName(mProgram.get(), cppSymbolName.data(), &loweredName));

        return CppLoweredSymbolName(loweredName, mProgram, CppLoweredSymbolName::PriviledgeToken{});
      }

      /**
       * @brief Get the code.
       * @param codeType The code type.
       * @return The code.
       */
      [[nodiscard]] Code getCode(CodeType codeType)
      {
        if (!mIsCompiled)
        {
          throw makeException<std::runtime_error>("The program is not compiled");
        }

        std::size_t size{};
        switch (codeType)
        {
        case CodeType::PTX:     Error::check(nvrtcGetPTXSize(mProgram.get(), &size));     break;
        case CodeType::CUBIN:   Error::check(nvrtcGetCUBINSize(mProgram.get(), &size));   break;
        case CodeType::LTOIR:   Error::check(nvrtcGetLTOIRSize(mProgram.get(), &size));   break;
        case CodeType::OptixIR: Error::check(nvrtcGetOptiXIRSize(mProgram.get(), &size)); break;
        default:
          throw makeException<std::runtime_error>("Invalid code type");
        }

        Code code(codeType, size, Code::PrivilegedToken{});
        switch (codeType)
        {
        case CodeType::PTX:     Error::check(nvrtcGetPTX(mProgram.get(), code.data()));     break;
        case CodeType::CUBIN:   Error::check(nvrtcGetCUBIN(mProgram.get(), code.data()));   break;
        case CodeType::LTOIR:   Error::check(nvrtcGetLTOIR(mProgram.get(), code.data()));   break;
        case CodeType::OptixIR: Error::check(nvrtcGetOptiXIR(mProgram.get(), code.data())); break;
        default:
          throw makeException<std::runtime_error>("Invalid code type");
        }

        return code;
      }

      [[nodiscard]] std::string_view getCompilationLog() const
      {
        return mCompilationLog;
      }
    protected:
    private:
      /**
       * @brief Add a name expression.
       * @param symbolName The name of the symbol.
       */
      void addNameExpression(std::string_view symbolName)
      {
        if (mIsCompiled)
        {
          throw makeException<std::runtime_error>("The program is already compiled");
        }

        Error::check(nvrtcAddNameExpression(mProgram.get(), symbolName.data()));
      }

      /**
       * @struct Deleter
       * @brief A deleter for the nvrtcProgram.
       */
      struct Deleter
      {
        /**
         * @brief Call operator.
         * @param program The program.
         */
        void operator()(nvrtcProgram program) const
        {
          if (program != nullptr)
          {
            nvrtcDestroyProgram(&program);
          }
        }
      };

      std::shared_ptr<std::remove_pointer_t<nvrtcProgram>> mProgram{};        ///< The program.
      bool                                                 mIsCompiled{};     ///< Program is compiled.
      std::string                                          mCompilationLog{}; ///< The compilation log.
  };

  /**
   * @brief Get the supported architectures.
   * @return The supported architectures.
   */
  [[nodiscard]] inline std::vector<int> getSupportedArchs()
  {
    int size{};

    Error::check(nvrtcGetNumSupportedArchs(&size));

    std::vector<int> archs(static_cast<std::size_t>(size));

    Error::check(nvrtcGetSupportedArchs(archs.data()));

    return archs;
  }

  /**
   * @brief Make an architecture option.
   * @param device The CUDA device.
   * @return The architecture option.
   */
  [[nodiscard]] inline std::string makeArchOption(int device)
  {
    if (!cuda::isValidDevice(device))
    {
      throw makeException<std::runtime_error>("Invalid device");
    }

    int ccMajor{};
    Error::check(cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, device));

    int ccMinor{};
    Error::check(cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, device));

    return cformat("-arch=sm_%d%d", ccMajor, ccMinor);
  }

  /**
   * @brief Make an include path option.
   * @param includePath The include path.
   * @return The include path option.
   */
  [[nodiscard]] inline std::string makeIncludePathOption(std::string_view includePath)
  {
    return cformat("-I%s", includePath.data());
  }

  /**
   * @brief Make a preprocessor definition option.
   * @param name The name of the definition.
   * @param value The value of the definition.
   * @return The preprocessor definition option.
   */
  [[nodiscard]] inline std::string makeDefinitionOption(std::string_view name, std::string_view value)
  {
    return cformat("-D%s=%s", name.data(), value.data());
  }

  /**
   * @brief Make a preprocessor definition option.
   * @param name The name of the definition.
   * @param value The value of the definition.
   * @return The preprocessor definition option.
   */
  [[nodiscard]] inline std::string makeDefinitionOption(std::string_view name)
  {
    return cformat("-D%s", name.data());
  }
  
  /**
   * @brief Make a debug option.
   * @return The debug option.
   */
  [[nodiscard]] inline std::string makeDebugOption()
  {
    return "-G";
  }
} // namespace afft::detail::cuda::rtc

#endif /* AFFT_DETAIL_CUDA_RTC_RTC_HPP */
