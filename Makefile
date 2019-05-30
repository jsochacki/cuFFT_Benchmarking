################################################################################
# Programmer:    John Sochacki
# Creation Date: May 23 2019
################################################################################

#**********************************************************************
# Preamble stuff - Skip down to Setup Project
#**********************************************************************
# Baseline empty, fill in with += later
CCPPFLAGS         =
NVCCFLAGS         =
INCLUDE           =
CPPDEPFLAGS       =
NVCCDEPFLAGS      =

# Output directories
MAKEROOT        := ./
OBJ             = $(MAKEROOT)obj/
GPUOBJ          = $(OBJ)GPU/
GPUOBJ_RL       = $(OBJ)reloc_GPU/
TMP             = $(MAKEROOT)tmp/

#**********************************************************************
# Setup Project
#**********************************************************************
# If debug is true you kernels will run SLOW (x10 slower) so dont turn on if you
# don't need it for debug
DEBUG = true
QUIET = true

# Benchmarking-related
BENCH_SRC_DIRS = $(sort $(dir $(wildcard ./benchmarks/*/)))

# Library directories
CUDA_DIR?=/usr/local/cuda
CUDA_BUILD_DIR=$(CUDA_DIR)/targets/x86_64-linux/

# Source/include directories and setup main target
SRCDIRS    =  ./src/ $(BENCH_SRC_DIRS)
INCDIRS    =  ./inc/ $(SRCDIRS) $(CUDA_DIR)/samples/common/inc
SYSINCDIRS =  $(CUDA_BUILD_DIR)include/

#**********************************************************************
# Object file creation and per-file dependency generation
#**********************************************************************
# vpath can't find auto-generated files, so we need to explicitly have a rule for them
vpath %.cpp $(SRCDIRS)
vpath %.cu $(SRCDIRS)
vpath %.c $(SRCDIRS)

TARGET      = $(MAKEROOT)executible
MAKEFILE    = $(MAKEROOT)Makefile

INCLUDE     += $(foreach DIR, $(INCDIRS), -I$(DIR)) $(foreach DIR, $(SYSINCDIRS), -isystem $(DIR))
CUINCLUDE   = -lcudart -lcublas -lcurand -lcufft

LADD        += -L$(CUDA_BUILD_DIR)lib -L$(OBJ) -L$(GPUOBJ) $(CUINCLUDE) -lrt -lm
#**********************************************************************
#
# Shouldn't need to touch anything below here...
#
#**********************************************************************
# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

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

TARGETSIZE  := -m${TARGET_SIZE}

NVCCFLAGS += $(TARGETSIZE)

# Debug build flags
ifeq ($(DEBUG), true)
		CCPPFLAGS += -g
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

# Gencode arguments
SMS ?= 30 35 37 50 52 60 61 70 75

ifeq ($(SMS),)
    $(error ERROR - no SM architectures have been specified!)
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
# For portability you want the opposite, to generate PTX from lowest possible arch
# so you would want arch=compute_highest code=comput_lowest
# as arch=compute_# is the virtual architecture and sets the __CUDA_ARCH__ def
# value when compiling while code=sm_# is the real architecture you are targeting
# lastly code=compute_# means generate PTX code for the runtime compiler that can
# be used if there is no binary (sm_) code available, there is only ever one
# code=compute as the resulting binary only keeps one
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

NVCCFLAGS+=$(GENCODE_FLAGS)


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
NVCC            = $(NVCCBIN)nvcc
NVCCCC          = -ccbin $(CCBIN)$(HOST_COMPILER)
CCPP            = $(CCBIN)$(HOST_COMPILER)
endif

ECHO            = @$(GNU)echo
RM              = $(GNU)rm -f
MKDEP           = @$(BIN)$(HOST_COMPILER) -MM -MG -MP
MKDIR           = @$(GNU)mkdir -p

ifeq ($(QUIET), true)
NVCC            := @$(NVCC)
CCPP            := @$(CCPP)
endif
#**********************************************************************
# Setup Targets
#**********************************************************************
all: $(TARGET)
	chmod a+x $(TARGET)

#**********************************************************************
# All objects
#**********************************************************************
# Rest of the automagic to find all c/cpp/cu files
FIND_CPP_FILES        = $(wildcard $(DIR)*.cpp)
FIND_CU_FILES         = $(wildcard $(DIR)*.cu)

CPP_FILES       := $(foreach DIR, $(SRCDIRS), $(FIND_CPP_FILES))
CU_FILES        := $(foreach DIR, $(SRCDIRS), $(FIND_CU_FILES))

CPP_FILE_OBJ    := $(patsubst %.cpp, $(OBJ)%.o, $(notdir $(CPP_FILES)))
CU_FILE_OBJ     := $(patsubst %.cu, $(GPUOBJ)%.o, $(notdir $(CU_FILES)))

CPP_FILE_DEP    := $(patsubst %.cpp, $(TMP)%.cpp.d, $(notdir $(CPP_FILES)))
CU_FILE_DEP     := $(patsubst %.cu, $(TMP)%.cu.d, $(notdir $(CU_FILES)))

DEPENDS         = $(CPP_FILE_DEP) $(CU_FILE_DEP)
OBJECTS         = $(CPP_FILE_OBJ)
GPUOBJECTS      = $(CU_FILE_OBJ)

$(OBJ)%.o: %.cpp $(MAKEFILE)
	$(ECHO) [$(notdir $<).d] from [$<]
	$(ECHO) -n $(OBJ) 	$(TMP)$(notdir $<).d
	$(MKDEP) $(CPPDEPFLAGS) $(INCLUDE) $< >	$(TMP)$(notdir $<).d
	$(ECHO) [$@] from [$<]
	$(NVCC) -x cu $(NVCCFLAGS) $(INCLUDE) -dc $(NVCCCC) $< -o $@

$(GPUOBJ)%.o: %.cu $(MAKEFILE)
	$(ECHO) [$(notdir $<).d] from [$<]
	$(ECHO) -n $(GPUOBJ) 	$(TMP)$(notdir $<).d
	$(MKDEP) $(NVCCDEPFLAGS) $(INCLUDE) $< >	$(TMP)$(notdir $<).d
	$(ECHO) [$@] from [$<]
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -dc $(NVCCCC) $< -o $@

$(GPUOBJ_RL)reloc_gpu_objects.o: $(OBJECTS) $(GPUOBJECTS) $(MAKEFILE)
	$(ECHO) [$@]
	$(NVCC) $(NVCCFLAGS) -dlink $(OBJECTS) $(GPUOBJECTS) -o $@

#**********************************************************************
# Object Linking and Target Output
#**********************************************************************

$(TARGET): $(OBJECTS) $(GPUOBJECTS) $(GPUOBJ_RL)reloc_gpu_objects.o
	$(ECHO) [$@]
	$(CCPP) $(CCPPFLAGS) $(GPUOBJ_RL)reloc_gpu_objects.o $(GPUOBJECTS) $(OBJECTS) $(LADD) -o $@

# MAKECMDGOALS is a make variable so you don't touch it, it is the target
ifeq (, $(filter $(MAKECMDGOALS), clean debug))
	# Things to execute if we aren't cleaning
 include $(TMP)tmpdir.txt
 include $(OBJ)objdir.txt
 include $(GPUOBJ)gpuobjdir.txt
 include $(GPUOBJ_RL)gpurelobjdir.txt
 -include $(DEPENDS)
endif

#**********************************************************************
# Makefile file generation
#**********************************************************************

$(TMP)tmpdir.txt:
	$(ECHO) [$@]
	$(MKDIR) $(dir $(TMP))
	$(ECHO) "# This is the Temporary File Directory." 	$@

$(OBJ)objdir.txt:
	$(ECHO) [$@]
	$(MKDIR) $(dir $(OBJ))
	$(ECHO) "# This is the OBJ File Directory." 	$@

$(GPUOBJ)gpuobjdir.txt:
	$(ECHO) [$@]
	$(MKDIR) $(dir $(GPUOBJ))
	$(ECHO) "# This is the GPUOBJ File Directory." 	$@

$(GPUOBJ_RL)gpurelobjdir.txt:
	$(ECHO) [$@]
	$(MKDIR) $(dir $(GPUOBJ_RL))
	$(ECHO) "# This is the GPUOBJ_RL File Directory." 	$@


.PHONY: debug
debug: $(MAKEFILE)
	$(ECHO) $(FIND_CPP_FILES)
	$(ECHO) $(FIND_CU_FILES)
	$(ECHO) $(CPP_FILES)
	$(ECHO) $(CU_FILES)
	$(ECHO) $(CPP_FILE_OBJ)
	$(ECHO) $(CU_FILE_OBJ)
	$(ECHO) $(CPP_FILE_DEP)
	$(ECHO) $(CU_FILE_DEP)
	$(ECHO) $(DEPENDS)
	$(ECHO) $(OBJECTS)
	$(ECHO) $(GPUOBJECTS)
	$(ECHO) $(NVCC)
	$(ECHO) $(CCPP)
	$(ECHO) $(CUDA_DIR)
	$(ECHO) $(NVCCBIN)
	$(ECHO) $(HOST_COMPILER)
	$(ECHO) $(TARGET_OS)


# Cleans up all temporary files, including object files.
.PHONY: clean
clean:
	$(ECHO)
	$(ECHO) Removing backup and temporary files
	$(ECHO)
	$(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR)*.bak))
	$(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR)*.~*))
	$(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR)*.tmp))
	$(RM) $(foreach DIR, $(SRCDIRS) $(MAKEROOT), $(wildcard $(DIR).*.swp))
	$(ECHO)
	$(ECHO) Removing object files
	$(ECHO)
	$(RM) $(notdir $(wildcard $(OBJ)*))
	$(RM) $(notdir $(wildcard $(GPUOBJ)*))
	$(RM) $(notdir $(wildcard $(GPUOBJ_RL)*))
	$(ECHO)
	$(ECHO) Removing linker output files
	$(ECHO)
	$(RM) $(TARGET)
	$(ECHO)
	$(ECHO) Removing makefile generated directories
	$(ECHO)
	$(RM) -r $(TMP)
	$(RM) -r $(OBJ)