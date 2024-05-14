# Compiler and flags (to be filled by the user)
CC = FCC
CFLAGS = -Nclang -Ofast -fopenmp -Nlibomp -ffj-swp -ffj-zfill -U__FJ_FIXEDLENGTH_SVE

# Source files
SRCS = main.cpp timer.cpp spmm_naive.cpp spmm_optimized.cpp utils.cpp

# Object files
OBJ_DIR = bin
OBJS = $(addprefix $(OBJ_DIR)/, $(SRCS:.cpp=.o))

# Target binary
TARGET = out

# Default target
all: $(TARGET)

# Compile source files into object files
# Objects files should be placed in the object directory
$(OBJ_DIR)/%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files into the target binary
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Clean the generated files
clean:
	rm -f $(OBJS) $(TARGET)
