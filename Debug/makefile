################################################################################
# Programmer:    John Sochacki
# Creation Date: May 23 2019
################################################################################

#**********************************************************************
# Preamble stuff - Skip down to Setup Project
#**********************************************************************
ifndef CONFIG
   UNAME := $(shell uname -s)
   ifeq ($(UNAME),Linux)
      CONFIG := LINUX
   endif
   ifeq ($(UNAME),Darwin)
      CONFIG := OSX
   endif
endif

ifndef CONFIG
   $(error Unable to detect operating system, please manually define CONFIG)
endif

# Baseline empty, fill in with += later
PROFILE           =
DEBUG             =
DEFINES           =
CCPPFLAGS         =
LFLAGS            =
LADD              =
CFLAGS            =
INCLUDE           =
CDEPFLAGS         =
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
COMPILER_CHAIN := COMPILE_CUDA_GCC
DPDK_TARGET = x86_64-native-linuxapp-gcc

INCDIRS=
SYSINCDIRS=

COTS_DIR=			../cots

# Architecture flags
ARCHITECTURE_TARGET=haswell

GCOV?=
COVERAGE?=

# Library directories

CUDA_DIR=/usr/local/cuda
CUDA_BUILD_DIR=$(CUDA_DIR)/targets/x86_64-linux/


SYSINCDIRS+=  $(CUDA_BUILD_DIR)include/


# Source/include directories and setup main target
SRCDIRS =  ./src/ ./src/utilities/ ./src/modules/ ./src/connectors/ ./src/transceivers/ ./src/workers/ ./src/workers/GPU/ ./unit_tests/ $(BENCH_SRC_DIRS) ./src/RPC/ ./src/RPC/services/ ./src/transceivers/BeaconDec/ ./src/transceivers/BeaconEnc/ ./src/transceivers/gfp/ ./src/transceivers/BondMux/ ./src/transceivers/TestFlMod/ ./src/utilities/GPU/
RELOC_SRCDIR = ./src/utilities/GPU/reloc/
INCDIRS+=  ./inc/ ./src/utilities/ ./src/connectors/ ./src/modules/ ./src/transceivers/ ./src/workers/ ./src/workers/GPU/ ./src/RPC $(PROTOS_AUTO_GENERATED_DIR) ./src/RPC/services/ ./src/RPC/interfaces/ ./src/transceivers/BeaconDec/ ./src/transceivers/BeaconEnc/ ./src/transceivers/gfp/ ./src/transceivers/BondMux/ ./src/transceivers/TestFlMod/ ./src/utilities/GPU/ ./src/utilities/GPU/reloc/

TARGET           = $(MAKEROOT)src/testbed
MAKEFILE         = $(MAKEROOT)Makefile

ASAN_FLAGS       = -fsanitize=address
ASAN_FLAGS       =
COMMON_CFLAGS    = -g -O2 -march=$(ARCHITECTURE_TARGET) $(ASAN_FLAGS)

DEBUG            +=
DEFINES          +=
LFLAGS           += $(ASAN_FLAGS) -std=c++14 -g -O2 -pthread -Wl,--whole-archive -Wl,-ldpdk -Wl,--start-group -Wl,--end-group -Wl,--no-whole-archive -libverbs -lnuma -lmnl $(GCOV) $(COVERAGE)
LADD             += -L$(BENCH_LIB) -L$(PB_LIB_DIR) -L$(GPRC_LIB_DIR) -L$(ZLIB_LIB_DIR) -L$(OPENSSL_LIB_DIR) -L$(DPDK_DIR)$(DPDK_TARGET)/lib -L$(CUDA_BUILD_DIR)lib -L$(OBJ) -L$(GPUOBJ) -L$(GTEST_LIB_DIR) -lcudart -lcublas -lcublasLt -lcurand -l:libcufft_static.a -l:libculibos.a -lnvToolsExt -l:libgtest.a -l:libconfig++.a -l:libprotobuf.a -l:libgrpc++_reflection.a -l:libgrpc++.a -l:libgrpc.a -lbenchmark -l:libz.a -l:libssl.a -l:libcrypto.a -lmlx5 -lrt -lm -ldl
CFLAGS           += $(COMMON_CFLAGS) $(COVERAGE)
CCPPFLAGS        += -std=c++17 $(COMMON_CFLAGS) $(COVERAGE)
NVCCFLAGS        += -g -lineinfo -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -std=c++14 --compiler-options -march=$(ARCHITECTURE_TARGET),-Wall,-Wno-unused-function,-fPIC,$(COVERAGE)
INCLUDE          += -include $(DPDK_DIR)$(DPDK_TARGET)/include/rte_config.h

CDEPFLAGS        +=
CPPDEPFLAGS      += -std=c++17 -march=$(ARCHITECTURE_TARGET)
NVCCDEPFLAGS     += -std=c++14 -march=$(ARCHITECTURE_TARGET)

all: $(TARGET)
	chmod a+x $(TARGET)

docker_dir=		../deploy/docker
matlab_dir=		./matlab


PARALLELISM = -j 32

#**********************************************************************
#
# Shouldn't need to touch anything below here...
#
#**********************************************************************

#**********************************************************************
# Setup Portable Directory Locations
#**********************************************************************

ifeq ($(CONFIG),CYGWIN)
   GNU             = /bin/
   CCBIN           = /usr/bin/
   BIN             = /usr/bin/
endif

ifeq ($(CONFIG),LINUX)
   GNU             = /bin/
   CCBIN           = /usr/bin/
   BIN             = /usr/bin/
   NVCCBIN         = $(CUDA_DIR)/bin/
endif

ifeq ($(CONFIG),OSX)
   GNU             = /bin/
   CCBIN           = /usr/bin/
   BIN             = /usr/bin/
   NVCCBIN         = $(CUDA_DIR)/bin/
endif

#**********************************************************************
# Setup Compiler and Binutils
#**********************************************************************

ifeq ($(COMPILER_CHAIN),COMPILE_CUDA_GCC)
CROSS_COMPILE   =
COMP_CC         = @$(CCBIN)$(CROSS_COMPILE)gcc
NVCC            = @$(NVCCBIN)$(CROSS_COMPILE)nvcc -ccbin $(CCBIN)$(CROSS_COMPILE)g++
LD              = @$(CCBIN)$(CROSS_COMPILE)g++
CCPP            = @$(CCBIN)$(CROSS_COMPILE)g++
endif

ECHO            = @$(GNU)echo
CURL            = @$(BIN)curl
RM              = $(GNU)rm -f
MV              = @$(GNU)mv
CP              = @$(GNU)cp
CPP             = @$(GNU)cpp
RMDIR           = -$(GNU)rmdir
MKDEP           = @$(BIN)$(CROSS_COMPILE)g++ -MM -MG -MP
MKDIR           = @$(GNU)mkdir -p
WGET            = @$(BIN)wget
INLINEWGET      = $(BIN)wget

#**********************************************************************
# Setup Compiler and Linker flags
#**********************************************************************

INCLUDE         += $(foreach DIR, $(INCDIRS), -I$(DIR)) $(foreach DIR, $(SYSINCDIRS), -isystem $(DIR))
DEBUG           +=
DEFINES         +=

LADD            +=

CCPPFLAGS       += $(DEFINES) $(DEBUG) $(INCLUDE)
LFLAGS          += $(PROFILE)
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
		$(ECHO)
		$(ECHO) Removing files in TEMP directory
		$(ECHO)
		$(RM) $(wildcard $(TMP)*)
		$(ECHO)
		$(ECHO) Removing makefile generated directories
		$(ECHO)
		$(RM) -r $(TMP)
		$(RM) -r $(OBJ)

#**********************************************************************
# All objects
#**********************************************************************

# Rest of the automagic to find all c/cpp/cu files
FIND_C_FILES          = $(wildcard $(DIR)*.c)
FIND_CPP_FILES        = $(wildcard $(DIR)*.cpp)
FIND_CU_FILES         = $(wildcard $(DIR)*.cu)

C_FILES         := $(filter-out $(patsubst %.pyx,%.c,$(PYX_FILES)), $(foreach DIR, $(SRCDIRS), $(FIND_C_FILES)))
CPP_FILES       := $(filter-out $(patsubst %.pyx,%.cpp,$(PYX_FILES)), $(foreach DIR, $(SRCDIRS), $(FIND_CPP_FILES))) \
                   $(GRPC_GRPC_PB_CPP_FILES) \
                   $(GRPC_PB_CPP_FILES)
CU_FILES        := $(foreach DIR, $(SRCDIRS), $(FIND_CU_FILES))
RELOC_CU_FILES  := $(foreach DIR, $(RELOC_SRCDIR), $(FIND_CU_FILES))

C_FILE_OBJ      := $(patsubst %.c, $(OBJ)%.o, $(notdir $(C_FILES)))
CPP_FILE_OBJ    := $(patsubst %.cpp, $(OBJ)%.o, $(notdir $(CPP_FILES)))
CU_FILE_OBJ     := $(patsubst %.cu, $(GPUOBJ)%.o, $(notdir $(CU_FILES)))
RELOC_CU_FILE_OBJ     := $(patsubst %.cu, $(GPUOBJ_RL)%.o, $(notdir $(RELOC_CU_FILES)))

C_FILE_DEP      := $(patsubst %.c, $(TMP)%.c.d, $(notdir $(C_FILES)))
CPP_FILE_DEP    := $(patsubst %.cpp, $(TMP)%.cpp.d, $(notdir $(CPP_FILES)))
CU_FILE_DEP     := $(patsubst %.cu, $(TMP)%.cu.d, $(notdir $(CU_FILES)))
CU_FILE_RL_DEP     := $(patsubst %.cu, $(TMP)%.cu.d, $(notdir $(RELOC_CU_FILES)))

DEPENDS         = $(C_FILE_DEP) $(CPP_FILE_DEP) $(CU_FILE_DEP) $(CU_FILE_RL_DEP)
OBJECTS         = $(C_FILE_OBJ) $(CPP_FILE_OBJ)
GPUOBJECTS      = $(CU_FILE_OBJ)
GPUOBJRELOC     = $(RELOC_CU_FILE_OBJ)

#**********************************************************************
# Object file creation and per-file dependency generation
#**********************************************************************
vpath %.cpp $(SRCDIRS)
vpath %.cu $(SRCDIRS)
vpath %.cu $(RELOC_SRCDIR)
vpath %.c $(SRCDIRS)
#vpath %.cpp $(PROTOS_AUTO_GENERATED_DIR)

# vpath can't find auto-generated files, so we need to explicitly have a rule for them
$(OBJ)%.o: $(PROTOS_AUTO_GENERATED_DIR)%.cpp $(MAKEFILE) $(COTS_DONE) $(TMP)protos $(ARTIFACTORY_DATA_DONE)
		$(ECHO) [$(notdir $<).d] from [$<]
		$(ECHO) -n $(OBJ) > $(TMP)$(notdir $<).d
		$(MKDEP) $(CPPDEPFLAGS) $< >> $(TMP)$(notdir $<).d
		$(ECHO) [$@] from [$<]
		$(CCPP) $(CCPPFLAGS) -c -o $@ $<

$(OBJ)%.o: %.c $(MAKEFILE) $(COTS_DONE) $(TMP)protos $(ARTIFACTORY_DATA_DONE)
		$(ECHO) [$(notdir $<).d] from [$<]
		$(ECHO) -n $(OBJ) > $(TMP)$(notdir $<).d
		$(MKDEP) $(CDEPFLAGS) $< >> $(TMP)$(notdir $<).d
		$(ECHO) [$@] from [$<]
		$(COMP_CC) $(CFLAGS) -c -o $@ $<

$(OBJ)%.o: %.cpp $(MAKEFILE) $(COTS_DONE) $(TMP)protos $(ARTIFACTORY_DATA_DONE)
		$(ECHO) [$(notdir $<).d] from [$<]
		$(ECHO) -n $(OBJ) > $(TMP)$(notdir $<).d
		$(MKDEP) $(CPPDEPFLAGS) $< >> $(TMP)$(notdir $<).d
		$(ECHO) [$@] from [$<]
		$(CCPP) $(CCPPFLAGS) -c -o $@ $<

$(GPUOBJ)%.o: %.cu $(MAKEFILE) $(COTS_DONE) $(TMP)protos $(ARTIFACTORY_DATA_DONE)
		$(ECHO) [$(notdir $<).d] from [non-reloc] [$<]
		$(ECHO) -n $(GPUOBJ) > $(TMP)$(notdir $<).d
		$(MKDEP) -x c++ $(NVCCDEPFLAGS) $< >> $(TMP)$(notdir $<).d
		$(ECHO) [$@] from [$<]
		$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(GPUOBJ_RL)%.o: %.cu $(MAKEFILE) $(COTS_DONE) $(TMP)protos $(ARTIFACTORY_DATA_DONE)
		$(ECHO) [$(notdir $<).d] from [reloc] [$<]
		$(ECHO) -n $(GPUOBJ_RL) > $(TMP)$(notdir $<).d
		$(MKDEP) -x c++ $(NVCCDEPFLAGS) $< >> $(TMP)$(notdir $<).d
		$(ECHO) [$@] from [$<]
		$(NVCC) $(NVCCFLAGS) -dc -o $@ $<
#**********************************************************************
# Makefile file generation
#**********************************************************************

$(TMP)tmpdir.txt:
		$(ECHO) [$@]
		$(MKDIR) $(dir $(TMP))
		$(ECHO) "# This is the Temporary File Directory." > $@

$(OBJ)objdir.txt:
		$(ECHO) [$@]
		$(MKDIR) $(dir $(OBJ))
		$(ECHO) "# This is the OBJ File Directory." > $@

$(GPUOBJ)gpuobjdir.txt:
		$(ECHO) [$@]
		$(MKDIR) $(dir $(GPUOBJ))
		$(ECHO) "# This is the GPUOBJ File Directory." > $@

$(GPUOBJ_RL)gpuobjdir_rl.txt:
		$(ECHO) [$@]
		$(MKDIR) $(dir $(GPUOBJ_RL))
		$(ECHO) "# This is the relocatable GPUOBJ File Directory." > $@

#**********************************************************************
# Object Linking and Target Output
#**********************************************************************

$(TARGET): $(OBJECTS) $(GPUOBJRELOC) $(GPUOBJECTS)
	$(ECHO) [$@]
	@git lfs status > /dev/null || (echo "git lfs files are bad: $$?.  Run 'git lfs pull'"; exit 1)
	$(NVCC) $(NVCCFLAGS) -dlink $(GPUOBJRELOC) -o $(GPUOBJ_RL)reloc_gpu_objects.o -lcudart -lcublas -lcurand -lcufft_static -lculibos -lnvToolsExt
	$(LD) $(LFLAGS) -o $@  $(GPUOBJ_RL)reloc_gpu_objects.o $(GPUOBJECTS) $(GPUOBJRELOC)  $(OBJECTS)  $(LADD)

ifeq (, $(filter $(MAKECMDGOALS), clean))
   # Things to execute if we aren't cleaning
   include $(TMP)tmpdir.txt
   include $(OBJ)objdir.txt
   include $(GPUOBJ_RL)gpuobjdir_rl.txt
   include $(GPUOBJ)gpuobjdir.txt

   -include $(DEPENDS)
endif



#####################REMOVE
# All Target
all: test

# Tool invocations
test: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: NVCC linker'
	/usr/local/cuda-10.0/bin/nvcc --cudart=static -ccbin g++ -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -o "test" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(EXECUTABLES)$(OBJS) test
	-@echo ' '

.PHONY: all clean dependents

-include ../makefile.targets
