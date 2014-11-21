CXX = g++
OBJ = obj
SRC = src
BIN = bin
LIB = lib
BINARY = Main
TESTS = tests
POINTC = $(wildcard $(SRC)*/*.c) $(wildcard $(SRC)/*.c) 
POINTCPP = $(wildcard $(SRC)/*/*/*.cpp) $(wildcard $(SRC)/*/*.cpp) $(wildcard $(SRC)/*.cpp) 
POINTOP := $(POINTC:.c=.o) $(POINTCPP:.cpp=.o)
POINTO = $(patsubst $(SRC)/%,$(OBJ)/%,$(POINTOP))

GLFWROOT = ../glfw/
GLMROOT = ../glm/
ANTTWEAKBARROOT = ../../Source/AntTweakBar/

ifeq ($(SHELL), sh.exe) 
OS := Win
else
OS := $(shell uname)
endif

ifeq ($(OS), Linux)
RM = rm
LIBS := -L "$(ANTTWEAKBARROOT)/lib" -lAntTweakBar -lGL
endif
ifeq ($(OS), Darwin)
RM = rm
LIBS := 
endif
ifeq ($(OS), Win)
RM = del
LIBS := -L "$(ANTTWEAKBARROOT)/lib" -lAntTweakBar -lglfw3 -lgdi32 -lopengl32 
endif

OPT := -std=c++11 -Wall -I "$(GLFWROOT)/include" -I "$(SRC)" -I "$(SRC)/Graphics" -I "$(SRC)/Graphics/ShaderProgram" -I "$(SRC)/Tools" -I "$(SRC)/Core" -I "$(GLMROOT)" -I "$(ANTTWEAKBARROOT)/include"

all : rel

debug : OPT := -g $(OPT)
debug : run

rel : OPT := -O3 $(OPT)
rel : run

.PHONY : dirs

windirs :
	mkdir bin
	mkdir obj
	mkdir obj\Graphics obj\Graphics\ShaderProgram obj\Tools obj\Core
	mkdir doc
	mkdir out
.PHONY : windirs

$(OBJ)/%.o : $(SRC)/%.cpp
	@echo --------------------------------------------------------------------------------
	@echo  [ Compilation of $^ ]
	$(CXX) $(OPT) $^ -c -o $@
	@echo --------------------------------------------------------------------------------
	
$(OBJ)/*/%.o : $(SRC)/*/%.cpp
	@echo --------------------------------------------------------------------------------
	@echo  [ Compilation of $^ ]
	$(CXX) $(OPT) $^ -c -o $@
	@echo --------------------------------------------------------------------------------
	
$(OBJ)/*/*/%.o : $(SRC)/*/*/%.cpp
	@echo --------------------------------------------------------------------------------
	@echo  [ Compilation of $^ ]
	$(CXX) $(OPT) $^ -c -o $@
	@echo --------------------------------------------------------------------------------

run : $(BINARY)
ifeq ($(OS), Win)
	$(BIN)\$(BINARY)
else
	./$(BIN)/$(BINARY)
endif
.PHONY : run

$(BINARY) : $(POINTO)
	@echo --------------------------------------------------------------------------------
	@echo  [ Linking $@ ]
	$(CXX) $(OPT) $^ -o $(BIN)/$@ $(LIBS)
	@echo --------------------------------------------------------------------------------
	
valgrind : $(BINARY)
	valgrind --leak-check=full --tool=memcheck ./$(BIN)/$(BINARY)
.PHONY : valgrind

clean:
ifeq ($(OS), Win)
	rd /s /q $(OBJ)
	rd /s /q $(BIN)
	make windirs
else
	$(RM) $(POINTO) $(BIN)/$(BINARY)
endif
.PHONY : clean

doc:
	doxygen Doxyfile
.PHONY : doc

