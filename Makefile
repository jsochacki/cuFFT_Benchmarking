################################################################################
# Programmer:    John Sochacki
# Creation Date: May 23 2019
################################################################################

.RECIPEPREFIX = >

#**********************************************************************
# Preamble stuff - Skip down to Setup Project
#**********************************************************************
# Baseline empty, fill in with += later
DEBUG             =
DEFINES           =
CCPPFLAGS         =
NVCCFLAGS         =
LFLAGS            =
LADD              =
CFLAGS            =
INCLUDE           =
CDEPFLAGS         =
CPPDEPFLAGS       =
NVCCDEPFLAGS      =

SRCDIRS=
INCDIRS=
SYSINCDIRS=

# Output directories
MAKEROOT        := ./
OBJ             = $(MAKEROOT)obj/
GPUOBJ          = $(OBJ)GPU/
GPUOBJ_RL       = $(OBJ)reloc_GPU/
TMP             = $(MAKEROOT)tmp/


#**********************************************************************
# Setup Project
#**********************************************************************
# 0 for no debug and 1 for debug
DBG = 0

# Architecture flags
ARCHITECTURE_TARGET=haswell

# Benchmarking-related
BENCH_SRC_DIRS = $(sort $(dir $(wildcard ./benchmarks/*/)))

# Library directories
CUDA_DIR?=/usr/local/cuda
CUDA_BUILD_DIR=$(CUDA_DIR)/targets/x86_64-linux/

# Source/include directories and setup main target
SRCDIRS+=  ./src/ $(BENCH_SRC_DIRS)
INCDIRS+=  ./inc/ $(CUDA_DIR)/samples/common/inc
SYSINCDIRS+=  $(CUDA_BUILD_DIR)include/

#**********************************************************************
# Object file creation and per-file dependency generation
#**********************************************************************
# vpath can't find auto-generated files, so we need to explicitly have a rule for them
vpath %.cpp $(SRCDIRS)
vpath %.cu $(SRCDIRS)
vpath %.c $(SRCDIRS)

TARGET           = $(MAKEROOT)executible
MAKEFILE         = $(MAKEROOT)Makefile

COMMON_CFLAGS    = -g -O2 -march=$(ARCHITECTURE_TARGET)

INCLUDE          += $(foreach DIR, $(INCDIRS), -I$(DIR)) $(foreach DIR, $(SYSINCDIRS), -isystem $(DIR))
CUINCLUDE        = -lcudart -lcublas -lcublasLt -lcurand -lcufft_static -lculibos -lnvToolsExt


DEBUG            +=
DEFINES          +=
LFLAGS           += -std=c++14 -g -O2 -pthread
LADD             += -L$(CUDA_BUILD_DIR)lib -L$(OBJ) -L$(GPUOBJ) $(CUINCLUDE) -lrt -lm
CFLAGS           += $(COMMON_CFLAGS)
CCPPFLAGS        += -std=c++14 $(COMMON_CFLAGS)
NVCCFLAGS        += -lineinfo -std=c++14 --compiler-options -march=$(ARCHITECTURE_TARGET),-Wall,-Wno-unused-function,-fPIC

CDEPFLAGS        +=
CPPDEPFLAGS      += -std=c++14 -march=$(ARCHITECTURE_TARGET)
NVCCDEPFLAGS     += -std=c++14 -march=$(ARCHITECTURE_TARGET)

all: $(TARGET)
	chmod a+x $(TARGET)

#**********************************************************************
#
# Shouldn't need to touch anything below here...
#
#**********************************************************************
##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif


# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-clang++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif

TARGETSIZE  := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

NVCCFLAGS+= $(TARGETSIZE)

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib -L $(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib -L $(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/aarch64-linux-gnu -L $(TARGET_FS)/usr/lib/aarch64-linux-gnu
            LDFLAGS += --unresolved-symbols=ignore-in-shared-libs
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include/aarch64-linux-gnu
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
        CCFLAGS += -DWIN_INTERFACE_CUSTOM -I/usr/include/aarch64-qnx-gnu
        LDFLAGS += -lsocket
        LDFLAGS += -rpath=/usr/lib/aarch64-qnx-gnu -L/usr/lib/aarch64-qnx-gnu
    endif
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif


# Debug build flags
ifeq ($(DBG),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))


# Gencode arguments
SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(SMS),)
    $(error ERROR - no SM architectures have been specified!)
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

#**********************************************************************
# Setup Portable Directory Locations
#**********************************************************************

ifeq ($(TARGET_OS),cygwin)
   GNU             = /bin/
   CCBIN           = /usr/bin/
   BIN             = /usr/bin/
endif

ifeq ($(TARGET_OS),linux)
   GNU             = /bin/
   CCBIN           = /usr/bin/
   BIN             = /usr/bin/
   NVCCBIN         = $(CUDA_DIR)/bin/
endif

ifeq ($(TARGET_OS),osx)
   GNU             = /bin/
   CCBIN           = /usr/bin/
   BIN             = /usr/bin/
   NVCCBIN         = $(CUDA_DIR)/bin/
endif

#**********************************************************************
# Setup Compiler and Binutils
#**********************************************************************

HOST_COMPILER ?= g++

ifeq ($(HOST_COMPILER),g++)
CROSS_COMPILE   =
COMP_CC         = @$(CCBIN)$(CROSS_COMPILE)gcc
NVCC            = @$(NVCCBIN)$(CROSS_COMPILE)nvcc -ccbin $(CCBIN)$(CROSS_COMPILE)$(HOST_COMPILER)
LD              = @$(CCBIN)$(CROSS_COMPILE)$(HOST_COMPILER)
CCPP            = @$(CCBIN)$(CROSS_COMPILE)$(HOST_COMPILER)
endif

ECHO            = @$(GNU)echo
CURL            = @$(BIN)curl
RM              = $(GNU)rm -f
MV              = @$(GNU)mv
CP              = @$(GNU)cp
CPP             = @$(GNU)cpp
RMDIR           = -$(GNU)rmdir
MKDEP           = @$(BIN)$(CROSS_COMPILE)$(HOST_COMPILER) -MM -MG -MP
MKDIR           = @$(GNU)mkdir -p
WGET            = @$(BIN)wget
INLINEWGET      = $(BIN)wget

#**********************************************************************
# Setup Compiler and Linker flags
#**********************************************************************

CCPPFLAGS       += $(DEFINES) $(DEBUG) $(INCLUDE)
CFLAGS          += $(DEFINES) $(DEBUG) $(INCLUDE)
NVCCFLAGS       += $(DEFINES) $(INCLUDE) # No DEBUG because NVCC doesn't support ICC flags
CDEPFLAGS       += $(DEFINES) $(INCLUDE)
CPPDEPFLAGS     += $(DEFINES) $(INCLUDE)
NVCCDEPFLAGS    += $(DEFINES) $(INCLUDE)

#**********************************************************************
# Setup Targets
#**********************************************************************

# Cleans up all temporary files, including object files.
.PHONY: clean
clean:
> $(ECHO)
> $(ECHO) Removing backup and temporary files
> $(ECHO)
> $(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR)*.bak))
> $(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR)*.~*))
> $(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR)*.tmp))
> $(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR).*.swp))
> $(ECHO)
> $(ECHO) Removing object files
> $(ECHO)
> $(RM) $(notdir $(wildcard $(OBJ)*))
> $(RM) $(notdir $(wildcard $(GPUOBJ)*))
> $(RM) $(notdir $(wildcard $(GPUOBJ_RL)*))
> $(ECHO)
> $(ECHO) Removing linker output files
> $(ECHO)
> $(RM) $(TARGET)
> $(ECHO)
> $(ECHO) Removing files in OBJ directory
> $(ECHO)
> $(RM) $(wildcard $(OBJ)*)
> $(ECHO)
> $(ECHO) Removing files in TEMP directory
> $(ECHO)
> $(RM) $(wildcard $(TMP)*)
> $(ECHO)
> $(ECHO) Removing makefile generated directories
> $(ECHO)
> $(RM) -r $(TMP)
> $(RM) -r $(OBJ)

#**********************************************************************
# All objects
#**********************************************************************

# Rest of the automagic to find all c/cpp/cu files
#Try removing DIR in the future as it is not defined and should not affect this
FIND_C_FILES          = $(wildcard $(DIR)*.c)
FIND_CPP_FILES        = $(wildcard $(DIR)*.cpp)
FIND_CU_FILES         = $(wildcard $(DIR)*.cu)

C_FILES         := $(foreach DIR, $(SRCDIRS), $(FIND_C_FILES))
CPP_FILES       := $(foreach DIR, $(SRCDIRS), $(FIND_CPP_FILES))
CU_FILES        := $(foreach DIR, $(SRCDIRS), $(FIND_CU_FILES))

C_FILE_OBJ      := $(patsubst %.c, $(OBJ)%.o, $(notdir $(C_FILES)))
CPP_FILE_OBJ    := $(patsubst %.cpp, $(OBJ)%.o, $(notdir $(CPP_FILES)))
CU_FILE_OBJ     := $(patsubst %.cu, $(GPUOBJ)%.o, $(notdir $(CU_FILES)))

C_FILE_DEP      := $(patsubst %.c, $(TMP)%.c.d, $(notdir $(C_FILES)))
CPP_FILE_DEP    := $(patsubst %.cpp, $(TMP)%.cpp.d, $(notdir $(CPP_FILES)))
CU_FILE_DEP     := $(patsubst %.cu, $(TMP)%.cu.d, $(notdir $(CU_FILES)))

DEPENDS         = $(C_FILE_DEP) $(CPP_FILE_DEP) $(CU_FILE_DEP)
OBJECTS         = $(C_FILE_OBJ) $(CPP_FILE_OBJ)
GPUOBJECTS      = $(CU_FILE_OBJ)


$(OBJ)%.o: $(MAKEFILE)
> $(ECHO) [$(notdir $<).d] from [$<]
> $(ECHO) -n $(OBJ) > $(TMP)$(notdir $<).d
> $(MKDEP) $(CPPDEPFLAGS) $< >> $(TMP)$(notdir $<).d
> $(ECHO) [$@] from [$<]
> $(CCPP) $(CCPPFLAGS) -c -o $@ $<

$(OBJ)%.o: %.c $(MAKEFILE)
> $(ECHO) [$(notdir $<).d] from [$<]
> $(ECHO) -n $(OBJ) > $(TMP)$(notdir $<).d
> $(MKDEP) $(CDEPFLAGS) $< >> $(TMP)$(notdir $<).d
> $(ECHO) [$@] from [$<]
> $(COMP_CC) $(CFLAGS) -c -o $@ $<

$(OBJ)%.o: %.cpp $(MAKEFILE)
> $(ECHO) [$(notdir $<).d] from [$<]
> $(ECHO) -n $(OBJ) > $(TMP)$(notdir $<).d
> $(MKDEP) $(CPPDEPFLAGS) $< >> $(TMP)$(notdir $<).d
> $(ECHO) [$@] from [$<]
> $(CCPP) $(CCPPFLAGS) -c -o $@ $<

$(GPUOBJ)%.o: %.cu $(MAKEFILE)
> $(ECHO) [$(notdir $<).d] from [$<]
> $(ECHO) -n $(GPUOBJ) > $(TMP)$(notdir $<).d
> $(MKDEP) -x c++ $(NVCCDEPFLAGS) $< >> $(TMP)$(notdir $<).d
> $(ECHO) [$@] from [$<]
> $(NVCC) $(NVCCFLAGS) -c -o $@ $<
#**********************************************************************
# Makefile file generation
#**********************************************************************

$(TMP)tmpdir.txt:
> $(ECHO) [$@]
> $(MKDIR) $(dir $(TMP))
> $(ECHO) "# This is the Temporary File Directory." > $@

$(OBJ)objdir.txt:
> $(ECHO) [$@]
> $(MKDIR) $(dir $(OBJ))
> $(ECHO) "# This is the OBJ File Directory." > $@

$(GPUOBJ)gpuobjdir.txt:
> $(ECHO) [$@]
> $(MKDIR) $(dir $(GPUOBJ))
> $(ECHO) "# This is the GPUOBJ File Directory." > $@

#**********************************************************************
# Object Linking and Target Output
#**********************************************************************

$(TARGET): $(OBJECTS) $(GPUOBJECTS)
> $(ECHO) [$@]
> $(NVCC) $(NVCCFLAGS) -dlink $(GPUOBJRELOC) -o $(GPUOBJ_RL)reloc_gpu_objects.o $(CUINCLUDE)
> $(LD) $(LFLAGS) -o $@  $(GPUOBJ_RL)reloc_gpu_objects.o $(GPUOBJECTS) $(GPUOBJRELOC)  $(OBJECTS)  $(LADD)

# MAKECMDGOALS is a make variable so you don't touch it, it is the target
ifeq (, $(filter $(MAKECMDGOALS), clean))
> # Things to execute if we aren't cleaning
> include $(TMP)tmpdir.txt
> include $(OBJ)objdir.txt
> include $(GPUOBJ)gpuobjdir.txt
> -include $(DEPENDS)
endif
